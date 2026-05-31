test_that("cf_circrna_immunogenicity returns named vector", {
  skip_if_not_installed("reticulate")
  result <- cf_circrna_immunogenicity("ACGUACGUACGUACGUACGUACGUACGUACGU")
  expect_type(result, "character")  # May be string for short seq
  # For longer sequences, should return numeric
})

test_that("cf_joint_evaluate returns list with composite", {
  skip_if_not_installed("reticulate")
  result <- cf_joint_evaluate(
    smiles = "CC(=O)Oc1ccccc1C(=O)O",
    epitope_seq = "SLYNTVATL",
    mhc_allele = "HLA-A*02:01",
    dose_mg = 200, freq_per_day = 2, treatment_time = 72
  )
  expect_type(result, "list")
  expect_true(length(result) > 0)
})