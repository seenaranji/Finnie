resource "aws_ecr_repository" "app" {
  name         = "finnie"
  force_delete = true # lets `terraform destroy` remove it even with images inside
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/finnie"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "main" { name = "finnie" }

# Execution role = what ECS itself needs to start the task (pull the image, read the secrets, write logs).
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "finnie-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "read_secrets" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [for s in aws_secretsmanager_secret.app : s.arn]
    }]
  })
}

locals {
  image = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  secret_env = [
    for name, env in {
      database_url          = "DATABASE_URL"
      jwt_secret            = "JWT_SECRET"
      openai_api_key        = "OPENAI_API_KEY"
      alpha_vantage_api_key = "ALPHA_VANTAGE_API_KEY"
    } : { name = env, valueFrom = aws_secretsmanager_secret.app[name].arn }
  ]
  log_options = { for k, v in {
    "awslogs-group"  = aws_cloudwatch_log_group.app.name
    "awslogs-region" = var.region
  } : k => v }
}

# One task, two containers that share localhost: the UI (public side) talks to the API at 127.0.0.1:8000. The API is never exposed.
resource "aws_ecs_task_definition" "app" {
  family                   = "finnie"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64" # build the image for linux/amd64 (see deploy/aws/README.md)
  }
  container_definitions = jsonencode([
    {
      name             = "api"
      image            = local.image
      essential        = true
      portMappings     = [{ containerPort = 8000 }]
      secrets          = local.secret_env
      logConfiguration = { logDriver = "awslogs", options = merge(local.log_options, { "awslogs-stream-prefix" = "api" }) }
      healthCheck = {
        command  = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
        interval = 15, timeout = 5, retries = 5, startPeriod = 30
      }
    },
    {
      name             = "ui"
      image            = local.image
      essential        = true
      command          = ["streamlit", "run", "src/web_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
      portMappings     = [{ containerPort = 8501 }]
      environment      = [{ name = "FINNIE_API_URL", value = "http://127.0.0.1:8000" }]
      dependsOn        = [{ containerName = "api", condition = "HEALTHY" }]
      logConfiguration = { logDriver = "awslogs", options = merge(local.log_options, { "awslogs-stream-prefix" = "ui" }) }
    }
  ])
}

# One-off jobs (migrate, index) reuse the same image and secrets; run them with `aws ecs run-task` (see README).
resource "aws_ecs_task_definition" "job" {
  family                   = "finnie-job"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name             = "job"
    image            = local.image
    essential        = true
    secrets          = local.secret_env
    logConfiguration = { logDriver = "awslogs", options = merge(local.log_options, { "awslogs-stream-prefix" = "job" }) }
  }])
}

resource "aws_ecs_service" "app" {
  name            = "finnie"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1 # one task only: Streamlit keeps session state in the process, so more tasks would need sticky sessions
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }
  health_check_grace_period_seconds = 60
  depends_on                        = [aws_lb_listener.http]
}
