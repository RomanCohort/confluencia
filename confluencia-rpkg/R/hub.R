#' Share a trained model with the community (federated)
#'
#' Uploads a .joblib model bundle to the Confluencia Hub.
#' Only the model weights are shared — raw training data is never exposed.
#' Set strip_env_medians=TRUE to also remove statistical traces of training data.
#'
#' @param bundle_path Path to the .joblib model bundle
#' @param metadata Named list of model metadata (e.g., list(model_name="ridge", r2=0.91))
#' @param uploader Uploader identifier (default: "anonymous")
#' @param license License for the shared model (default: "MIT")
#' @param strip_env_medians Remove env_medians to eliminate training data traces (default: FALSE)
#' @return Named list with model_id and message
#' @export
#' @examples
#' \dontrun{
#' # Share with full privacy: strip statistical traces
#' result <- cf_hub_push_model("my_drug_bundle.joblib",
#'                              metadata = list(model_name = "ridge", r2 = 0.91),
#'                              strip_env_medians = TRUE)
#' print(result$model_id)
#' }
cf_hub_push_model <- function(bundle_path, metadata = NULL,
                               uploader = "anonymous", license = "MIT",
                               strip_env_medians = FALSE) {
  bridge <- .cf_bridge()
  result <- bridge$hub_push_model(
    bundle_path = bundle_path,
    metadata = metadata,
    uploader = uploader,
    license = license,
    strip_env_medians = strip_env_medians
  )
  as.list(result)
}

#' Download a community model from the hub
#'
#' @param model_id Model identifier (e.g., "hub:drug:anonymous:abc123")
#' @return Named list with bundle_path (local path to downloaded .joblib)
#' @export
cf_hub_pull_model <- function(model_id) {
  bridge <- .cf_bridge()
  result <- bridge$hub_pull_model(model_id = model_id)
  as.list(result)
}

#' List available community models
#'
#' @param task Filter by task: "drug" or "epitope" (NULL for all)
#' @param limit Maximum number of models to return (default: 50)
#' @return Data frame of available models with metadata
#' @export
cf_hub_list_models <- function(task = NULL, limit = 50L) {
  bridge <- .cf_bridge()
  result <- bridge$hub_list_models(task = task, limit = as.integer(limit))
  if (length(result) == 0) return(data.frame())
  do.call(rbind, lapply(result, function(x) as.data.frame(x, stringsAsFactors = FALSE)))
}

#' Contribute a dataset to the community pool
#'
#' Uploads a CSV to the Confluencia Hub. By default, the upload is anonymous.
#' You must specify a license (CC-BY-4.0, MIT, or proprietary).
#'
#' @param csv_path Path to the CSV file
#' @param license Data license (default: "CC-BY-4.0")
#' @param anonymous Whether to upload anonymously (default: TRUE)
#' @param uploader Uploader identifier (default: "anonymous")
#' @return Named list with dataset_id and message
#' @export
#' @examples
#' \dontrun{
#' result <- cf_hub_push_data("my_drug_data.csv", license = "CC-BY-4.0")
#' print(result$dataset_id)
#' }
cf_hub_push_data <- function(csv_path, license = "CC-BY-4.0",
                               anonymous = TRUE, uploader = "anonymous") {
  bridge <- .cf_bridge()
  result <- bridge$hub_push_data(
    csv_path = csv_path,
    license = license,
    anonymous = anonymous,
    uploader = uploader
  )
  as.list(result)
}

#' Get community dataset statistics
#'
#' Returns aggregate statistics about community datasets.
#' No raw data or individual records are exposed.
#'
#' @return Named list with per-task sample counts and contributor counts
#' @export
cf_hub_data_stats <- function() {
  bridge <- .cf_bridge()
  result <- bridge$hub_data_stats()
  as.list(result)
}
