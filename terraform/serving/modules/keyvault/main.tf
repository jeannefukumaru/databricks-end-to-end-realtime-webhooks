data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "kv" {
  name                        = var.name
  location                    = var.location
  resource_group_name         = var.resource_group_name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false

  sku_name = var.sku_name

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Set",
      "Get",
      "List",
      "Delete",
      "Purge"
    ]
  }
}

resource "azurerm_key_vault_secret" "kube_config" {
  name         = "kubeconfig"
  value        = var.secret_kube_config
  key_vault_id = azurerm_key_vault.kv.id
}

resource "azurerm_key_vault_secret" "kube_certificate" {
  name         = "kubecertificate"
  value        = var.secret_kube_certificate
  key_vault_id = azurerm_key_vault.kv.id
}