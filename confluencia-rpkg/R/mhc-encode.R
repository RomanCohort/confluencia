#' Encode MHC features for a peptide-allele pair
#'
#' @param peptide Peptide sequence string
#' @param allele MHC allele string (e.g., "HLA-A*02:01")
#' @return Numeric vector (1018-dim for MHC-I, 947-dim for MHC-II)
#' @export
cf_mhc_encode <- function(peptide, allele) {
  bridge <- .cf_bridge()
  result <- bridge$mhc_encode(peptide = peptide, allele = allele)
  unlist(result)
}

#' Detect MHC class from allele string
#'
#' @param allele MHC allele string
#' @return Character: "I" or "II"
#' @export
cf_mhc_detect_class <- function(allele) {
  bridge <- .cf_bridge()
  as.character(bridge$mhc_detect_class(allele = allele))
}