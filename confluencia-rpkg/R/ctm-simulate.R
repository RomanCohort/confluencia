#' Derive CTM parameters from micro scores
#'
#' Maps binding, immune, and inflammation scores to CTMParams
#' for small-molecule PK simulation.
#'
#' @param binding Binding affinity score (0-1)
#' @param immune Immune activation score (0-1)
#' @param inflammation Inflammation score (0-1)
#' @return Named numeric vector with CTM parameters (ka, kd, ke, km, signal_gain)
#' @export
#' @examples
#' cf_ctm_params(binding = 0.72, immune = 0.65, inflammation = 0.12)
cf_ctm_params <- function(binding = 0.5, immune = 0.5, inflammation = 0.5) {
  bridge <- .cf_bridge()
  result <- bridge$ctm_params(
    binding = as.numeric(binding),
    immune = as.numeric(immune),
    inflammation = as.numeric(inflammation)
  )
  unlist(as.list(result))
}

#' Simulate small-molecule PK (CTM 6-compartment model)
#'
#' Simulates the 6-compartment pharmacokinetic model for small molecules.
#'
#' @param dose Dose in mg
#' @param freq Dosing frequency per day
#' @param params Named numeric vector from cf_ctm_params(), or NULL to auto-derive
#' @param binding Binding affinity score (used if params is NULL)
#' @param immune Immune activation score (used if params is NULL)
#' @param inflammation Inflammation score (used if params is NULL)
#' @param horizon Simulation horizon in hours
#' @return data.frame with columns: time_h, absorption_A, distribution_D,
#'   effect_E, metabolism_M, efficacy_signal, toxicity_signal
#' @export
#' @examples
#' cf_ctm_simulate(dose = 200, freq = 2, binding = 0.72, immune = 0.65)
cf_ctm_simulate <- function(dose, freq, params = NULL,
                             binding = 0.5, immune = 0.5, inflammation = 0.5,
                             horizon = 72) {
  bridge <- .cf_bridge()
  params_dict <- if (is.null(params)) NULL else as.list(params)
  result <- bridge$ctm_simulate(
    dose = as.numeric(dose),
    freq = as.numeric(freq),
    params_dict = params_dict,
    binding = as.numeric(binding),
    immune = as.numeric(immune),
    inflammation = as.numeric(inflammation),
    horizon = as.integer(horizon)
  )
  .py_dict_to_df(result)
}

#' Derive RNA-CTM parameters from modification and delivery config
#'
#' @param modification Nucleotide modification type: "none", "m6A", "pseudouridine", "5mC", "ms2m6A"
#' @param delivery_vector Delivery vector type: "LNP_standard", "LNP_liver", "AAV", "naked"
#' @param route Administration route: "IV", "SC", "IM"
#' @param ires_score IRES efficiency score (0-1)
#' @param gc_content GC content (0-1)
#' @param struct_stability Structure stability score (0-1)
#' @param innate_immune_score Innate immune activation score (0-1)
#' @return Named numeric vector with RNA-CTM parameters
#' @export
cf_rna_ctm_params <- function(modification = "none", delivery_vector = "LNP_standard",
                               route = "IV", ires_score = 0.5, gc_content = 0.5,
                               struct_stability = 0.5, innate_immune_score = 0.0) {
  bridge <- .cf_bridge()
  result <- bridge$rna_ctm_params(
    modification = modification, delivery_vector = delivery_vector,
    route = route, ires_score = as.numeric(ires_score),
    gc_content = as.numeric(gc_content),
    struct_stability = as.numeric(struct_stability),
    innate_immune_score = as.numeric(innate_immune_score)
  )
  unlist(as.list(result))
}

#' Simulate circRNA PK (RNA-CTM 6-compartment model)
#'
#' @param dose Dose in mg
#' @param freq Dosing frequency per day
#' @param params Named numeric vector from cf_rna_ctm_params(), or NULL
#' @param modification Nucleotide modification (used if params is NULL)
#' @param delivery_vector Delivery vector (used if params is NULL)
#' @param route Administration route (used if params is NULL)
#' @param horizon Simulation horizon in hours
#' @return data.frame with multi-compartment PK columns
#' @export
cf_rna_ctm_simulate <- function(dose, freq, params = NULL,
                                 modification = "none", delivery_vector = "LNP_standard",
                                 route = "IV", horizon = 168) {
  bridge <- .cf_bridge()
  params_dict <- if (is.null(params)) NULL else as.list(params)
  result <- bridge$rna_ctm_simulate(
    dose = as.numeric(dose), freq = as.numeric(freq),
    params_dict = params_dict, modification = modification,
    delivery_vector = delivery_vector, route = route,
    horizon = as.integer(horizon)
  )
  .py_dict_to_df(result)
}