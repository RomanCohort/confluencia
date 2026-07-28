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
#' @param uploader_orcid Canonical ORCID iD (e.g., "0000-0002-1825-0097"). Strongly
#'   recommended — binds the upload to a citable identity and enables Zenodo DOI minting.
#' @param contributors Character vector of additional ORCID iDs of co-contributors
#' @param reproducibility_url URL to training code repo (raises tier to "reproducible")
#' @param mint_doi If TRUE (default) and CONFLUENCIA_ZENODO_TOKEN is set, mint a DOI
#' @return Named list with model_id and message
#' @export
#' @examples
#' \dontrun{
#' # Share with full privacy: strip statistical traces
#' result <- cf_hub_push_model("my_drug_bundle.joblib",
#'                              metadata = list(model_name = "ridge", r2 = 0.91),
#'                              strip_env_medians = TRUE,
#'                              uploader_orcid = "0000-0002-1825-0097")
#' print(result$model_id)
#' }
cf_hub_push_model <- function(bundle_path, metadata = NULL,
                               uploader = "anonymous", license = "MIT",
                               strip_env_medians = FALSE,
                               uploader_orcid = "",
                               contributors = NULL,
                               reproducibility_url = "",
                               mint_doi = TRUE) {
  bridge <- .cf_bridge()
  result <- bridge$hub_push_model(
    bundle_path = bundle_path,
    metadata = metadata,
    uploader = uploader,
    license = license,
    strip_env_medians = strip_env_medians,
    uploader_orcid = uploader_orcid,
    contributors = contributors %||% list(),
    reproducibility_url = reproducibility_url,
    mint_doi = mint_doi
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

#' Aggregate impact report for a contributor (by ORCID)
#'
#' Returns total downloads, citations, badge, and per-model breakdown for a
#' contributor. Useful for annual reports / CV entries.
#'
#' @param orcid Contributor ORCID iD (e.g., "0000-0002-1825-0097")
#' @return Named list: orcid, n_models, total_downloads, total_citations, badge, models
#' @export
#' @examples
#' \dontrun{
#' report <- cf_hub_contributor_stats("0000-0002-1825-0097")
#' print(report$total_downloads)
#' }
cf_hub_contributor_stats <- function(orcid) {
  bridge <- .cf_bridge()
  result <- bridge$hub_get_contributor_stats(orcid = orcid)
  as.list(result)
}

#' Promote a model's verification tier
#'
#' Tiers (low -> high): unverified < reproducible < verified < benchmark_top.
#' A tier can only be raised, never lowered, by this method.
#'
#' @param model_id The hub model ID to verify
#' @param level Target tier: "reproducible", "verified", or "benchmark_top"
#' @param reviewer ORCID or name of the verifier (for audit trail)
#' @param evidence_url URL supporting the claim (e.g., Circ-CASP results page)
#' @param circ_casp_metrics Named list of competition metrics (e.g., list(rmsd=8.2, rank=3))
#' @return Named list with updated verification_level and audit info
#' @export
#' @examples
#' \dontrun{
#' cf_hub_verify_model("hub:circRNA:0000-0002-1825-0097:abc123def456",
#'                     level = "verified",
#'                     reviewer = "circ-casp-committee",
#'                     circ_casp_metrics = list(rmsd = 8.2, rank = 3))
#' }
cf_hub_verify_model <- function(model_id, level,
                                  reviewer = "",
                                  evidence_url = "",
                                  circ_casp_metrics = NULL) {
  bridge <- .cf_bridge()
  result <- bridge$hub_verify_model(
    model_id = model_id,
    level = level,
    reviewer = reviewer,
    evidence_url = evidence_url,
    circ_casp_metrics = circ_casp_metrics %||% list()
  )
  as.list(result)
}

#' Upload a Circ-CASP competition model with auto-bound results
#'
#' Wraps cf_hub_push_model for the competition flow: task forced to "circRNA",
#' circ_casp_metrics bound to metadata, verification starts at "reproducible"
#' if code repo provided.
#'
#' @param bundle_path Path to the .joblib model bundle
#' @param uploader_orcid Team lead ORCID (required for DOI minting)
#' @param team_name Circ-CASP team name
#' @param circ_casp_metrics Named list of competition results, e.g.:
#'   list(rmsd=8.2, t1=80, t2=100, t3=75, t4=60, t5=50, total=71.5, rank=3)
#' @param reproducibility_url Inference code repo (raises tier to reproducible)
#' @param contributors Character vector of additional team member ORCIDs
#' @param license Model weight license (default: "MIT")
#' @return Named list with model_id of the uploaded competition model
#' @export
#' @examples
#' \dontrun{
#' cf_hub_push_circ_casp_submission(
#'   "my_circ_model.joblib",
#'   uploader_orcid = "0000-0002-1825-0097",
#'   team_name = "TorusFold-X",
#'   circ_casp_metrics = list(rmsd = 8.2, total = 71.5, rank = 3),
#'   reproducibility_url = "https://github.com/team/torusfold-x"
#' )
#' }
cf_hub_push_circ_casp_submission <- function(bundle_path, uploader_orcid,
                                               team_name, circ_casp_metrics,
                                               reproducibility_url = "",
                                               contributors = NULL,
                                               license = "MIT") {
  bridge <- .cf_bridge()
  result <- bridge$hub_push_circ_casp_submission(
    bundle_path = bundle_path,
    uploader_orcid = uploader_orcid,
    team_name = team_name,
    circ_casp_metrics = circ_casp_metrics,
    reproducibility_url = reproducibility_url,
    contributors = contributors %||% list(),
    license = license
  )
  as.list(result)
}

#' List circRNA models eligible as Circ-CASP baselines
#'
#' A model is baseline-eligible if its verification_level >= min_tier.
#' Benchmark-top models are surfaced first.
#'
#' @param min_tier Minimum verification tier: "verified" (default) or "benchmark_top"
#' @param limit Maximum number of models to return (default: 20)
#' @return Data frame of eligible baseline models
#' @export
cf_hub_list_circ_casp_baselines <- function(min_tier = "verified", limit = 20L) {
  bridge <- .cf_bridge()
  result <- bridge$hub_list_circ_casp_baselines(
    min_tier = min_tier, limit = as.integer(limit)
  )
  if (length(result) == 0) return(data.frame())
  do.call(rbind, lapply(result, function(x) as.data.frame(x, stringsAsFactors = FALSE)))
}
