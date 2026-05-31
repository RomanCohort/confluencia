#' Predict circRNA immunogenicity scores
#'
#' Predicts RIG-I, TLR7/8, and PKR pathway activation scores
#' for a circRNA sequence.
#'
#' @param sequence RNA sequence string (A, C, G, U)
#' @return Named numeric vector with immune pathway scores
#' @export
#' @examples
#' cf_circrna_immunogenicity("ACGUACGUACGUACGUACGUACGUACGUACGU")
cf_circrna_immunogenicity <- function(sequence) {
  bridge <- .cf_bridge()
  result <- bridge$circrna_immunogenicity(sequence = sequence)
  .py_dict_to_list(result)
}

#' Run full circRNA pipeline
#'
#' @param sequence RNA sequence string
#' @param gene_expression Named list of gene expression values
#' @return List with immune_scores, composite_scores, recommendations, uncertainty
#' @export
cf_circrna_pipeline <- function(sequence, gene_expression = list()) {
  bridge <- .cf_bridge()
  ge <- if (length(gene_expression) == 0) NULL else gene_expression
  result <- bridge$circrna_pipeline(
    sequence = sequence, gene_expression = ge
  )
  as.list(result)
}