#' Run 5D joint evaluation
#'
#' Evaluates a drug-epitope-circRNA candidate across five dimensions:
#' clinical efficacy, target binding, kinetics, gene signature, and circRNA.
#'
#' @param smiles SMILES molecular string
#' @param epitope_seq Peptide amino acid sequence
#' @param mhc_allele MHC allele string
#' @param dose_mg Dose in mg
#' @param freq_per_day Dosing frequency per day
#' @param treatment_time Treatment duration in hours
#' @param circ_expr circRNA expression level (0-1)
#' @param ifn_score Interferon score (0-1)
#' @param group_id Group identifier string
#' @param trop2 TROP2 expression (0-1)
#' @param nectin4 Nectin-4 expression (0-1)
#' @param liv1 LIV-1 expression (0-1)
#' @param b7h4 B7-H4 expression (0-1)
#' @param tmem65 TMEM65 expression (0-1)
#' @param circ_sequence Optional circRNA sequence
#' @return List with composite score, recommendation, individual dimension scores
#' @export
cf_joint_evaluate <- function(smiles, epitope_seq, mhc_allele,
                               dose_mg, freq_per_day, treatment_time,
                               circ_expr = 0, ifn_score = 0,
                               group_id = "G0",
                               trop2 = 0.5, nectin4 = 0.5,
                               liv1 = 0.5, b7h4 = 0.5, tmem65 = 0.5,
                               circ_sequence = NULL) {
  bridge <- .cf_bridge()
  input_dict <- list(
    smiles = smiles, epitope_seq = epitope_seq,
    mhc_allele = mhc_allele, dose_mg = as.numeric(dose_mg),
    freq_per_day = as.numeric(freq_per_day),
    treatment_time = as.numeric(treatment_time),
    circ_expr = as.numeric(circ_expr), ifn_score = as.numeric(ifn_score),
    group_id = group_id, trop2 = as.numeric(trop2),
    nectin4 = as.numeric(nectin4), liv1 = as.numeric(liv1),
    b7h4 = as.numeric(b7h4), tmem65 = as.numeric(tmem65)
  )
  if (!is.null(circ_sequence)) {
    input_dict$circ_sequence <- circ_sequence
  }
  result <- bridge$joint_evaluate(input_dict = input_dict)
  as.list(result)
}