# Values live in Secrets Manager and are injected into the containers at start. They are NOT in the image or in the task definition text.
# Note: Terraform keeps these values in its state file too, so keep terraform.tfstate private (it is git-ignored) or use a remote encrypted backend.
resource "aws_secretsmanager_secret" "app" {
  for_each                = toset(["database_url", "jwt_secret", "openai_api_key", "alpha_vantage_api_key"])
  name                    = "finnie/${each.key}"
  recovery_window_in_days = 0 # delete immediately on destroy so the name can be reused
}

locals {
  secret_values = {
    database_url          = "postgresql://${aws_db_instance.main.username}:${random_password.db.result}@${aws_db_instance.main.address}:5432/${aws_db_instance.main.db_name}"
    jwt_secret            = random_password.jwt.result
    openai_api_key        = var.openai_api_key
    alpha_vantage_api_key = var.alpha_vantage_api_key == "" ? "unset" : var.alpha_vantage_api_key # Secrets Manager refuses an empty value
  }
}

resource "aws_secretsmanager_secret_version" "app" {
  for_each      = aws_secretsmanager_secret.app
  secret_id     = each.value.id
  secret_string = local.secret_values[each.key]
}
