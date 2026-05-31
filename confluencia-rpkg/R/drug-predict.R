#' Predict drug efficacy for a single SMILES
#'
#' @param bundle_path Path to a trained .joblib DrugModelBundle
#' @param smiles SMILES string
#' @param env_params Named list of environmental parameters (e.g., list(conc = 5.0))
#' @return Numeric scalar (predicted efficacy)
#' @export
cf_drug_predict <- function(bundle_path, smiles, env_params = list()) {
  bridge <- .cf_bridge()
  ep <- .validate_env_params(env_params)
  result <- bridge$drug_predict(
    bundle_path = bundle_path, smiles = smiles,
    env_params = if (length(ep) == 0) NULL else ep
  )
  as.numeric(result)
}

#' Predict epitope binding for a single sequence
#'
#' @param bundle_path Path to a trained .joblib EpitopeModelBundle
#' @param sequence Peptide amino acid sequence
#' @param env_params Named list of environmental parameters
#' @return Numeric scalar (predicted binding)
#' @export
cf_epitope_predict <- function(bundle_path, sequence, env_params = list()) {
  bridge <- .cf_bridge()
  ep <- .validate_env_params(env_params)
  result <- bridge$epitope_predict(
    bundle_path = bundle_path, sequence = sequence,
    env_params = if (length(ep) == 0) NULL else ep
  )
  as.numeric(result)
}