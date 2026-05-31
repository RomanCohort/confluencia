#' Register a custom model type
#'
#' Register a model class from any Python package so it can be used with
#' cf_drug_train() and cf_epitope_train().
#'
#' @param name Model name (e.g., "xgboost")
#' @param module_path Python module path (e.g., "xgboost")
#' @param function_name Python class/function name (e.g., "XGBRegressor")
#' @return Named list with name and message
#' @export
#' @examples
#' \dontrun{
#' # Register XGBoost for use in training
#' cf_register_model("xgboost", "xgboost", "XGBRegressor")
#' result <- cf_drug_train("data.csv", model_name = "xgboost")
#' }
cf_register_model <- function(name, module_path, function_name) {
  bridge <- .cf_bridge()
  result <- bridge$plugin_register_model(
    name = name,
    module_path = module_path,
    function_name = function_name
  )
  as.list(result)
}

#' Register a custom sequence encoder
#'
#' @param name Encoder name (e.g., "esm2_small")
#' @param module_path Python module path
#' @param function_name Python function name
#' @return Named list with name and message
#' @export
cf_register_encoder <- function(name, module_path, function_name) {
  bridge <- .cf_bridge()
  result <- bridge$plugin_register_encoder(
    name = name,
    module_path = module_path,
    function_name = function_name
  )
  as.list(result)
}

#' Register a new evaluation dimension
#'
#' Add a custom dimension to the 5D evaluation framework.
#' After registration, it will be included in cf_joint_evaluate() results.
#'
#' @param name Dimension name (e.g., "manufacturability")
#' @param weight Weight in composite score (0-1)
#' @param description Human-readable description
#' @return Named list with name, weight, and message
#' @export
#' @examples
#' \dontrun{
#' cf_register_dimension("manufacturability", weight = 0.1,
#'                        description = "Ease of large-scale production")
#' }
cf_register_dimension <- function(name, weight, description = "") {
  bridge <- .cf_bridge()
  result <- bridge$plugin_register_dimension(
    name = name,
    weight = as.numeric(weight),
    description = description
  )
  as.list(result)
}

#' Set scoring weights for evaluation dimensions
#'
#' Modify the weights used to compute the composite score in cf_joint_evaluate().
#' Weights are normalized to sum to 1.0.
#'
#' @param weights Named numeric vector of dimension weights
#' @return Named list with normalized weights
#' @export
#' @examples
#' \dontrun{
#' cf_set_weights(c(clinical = 0.30, binding = 0.25, kinetics = 0.15,
#'                  gene_signature = 0.15, circrna = 0.15))
#' }
cf_set_weights <- function(weights) {
  bridge <- .cf_bridge()
  result <- bridge$plugin_set_weights(weights = as.list(weights))
  result
}

#' List all registered plugins
#'
#' Returns all registered models, encoders, PK solvers, evaluation dimensions,
#' and current scoring weights.
#'
#' @return Named list with models, encoders, pk_solvers, dimensions, weights
#' @export
cf_list_plugins <- function() {
  bridge <- .cf_bridge()
  result <- bridge$plugin_list()
  result
}
