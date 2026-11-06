# Managed PostgreSQL. RDS supports the pgvector extension; migration 0001 runs CREATE EXTENSION vector (the master user may do that).
resource "random_password" "db" {
  length  = 32
  special = false # keeps the password safe to place inside a URL
}

resource "random_password" "jwt" {
  length  = 64
  special = false
}

resource "aws_db_subnet_group" "main" {
  name       = "finnie"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "main" {
  identifier                   = "finnie"
  engine                       = "postgres"
  engine_version               = "17"
  instance_class               = var.db_instance_class
  allocated_storage            = 20
  storage_type                 = "gp3"
  storage_encrypted            = true
  db_name                      = "finnie"
  username                     = "finnie"
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.main.name
  vpc_security_group_ids       = [aws_security_group.db.id]
  publicly_accessible          = false
  multi_az                     = false # true doubles the cost; turn on for real use
  backup_retention_period      = 7
  deletion_protection          = var.deletion_protection
  skip_final_snapshot          = !var.deletion_protection
  final_snapshot_identifier    = var.deletion_protection ? "finnie-final" : null
  auto_minor_version_upgrade   = true
  performance_insights_enabled = false
}
