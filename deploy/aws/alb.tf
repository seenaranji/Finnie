resource "aws_lb" "main" {
  name               = "finnie"
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
  idle_timeout       = 120 # Streamlit holds a long-lived websocket
}

resource "aws_lb_target_group" "ui" {
  name        = "finnie-ui"
  port        = 8501
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id
  health_check {
    path    = "/_stcore/health"
    matcher = "200"
  }
  stickiness {
    type    = "lb_cookie"
    enabled = true
  }
}

locals { https = var.certificate_arn != "" }

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = local.https ? "redirect" : "forward"
    target_group_arn = local.https ? null : aws_lb_target_group.ui.arn
    dynamic "redirect" {
      for_each = local.https ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = local.https ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}
