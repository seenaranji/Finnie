output "url" { value = "${local.https ? "https" : "http"}://${aws_lb.main.dns_name}" }
output "ecr_repository" { value = aws_ecr_repository.app.repository_url }
output "cluster" { value = aws_ecs_cluster.main.name }
output "job_task_definition" { value = aws_ecs_task_definition.job.family }
output "public_subnet_ids" { value = aws_subnet.public[*].id }
output "task_security_group" { value = aws_security_group.task.id }
