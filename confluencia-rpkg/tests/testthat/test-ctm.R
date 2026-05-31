test_that("cf_ctm_params returns named numeric vector", {
  skip_if_not_installed("reticulate")
  params <- cf_ctm_params(binding = 0.72, immune = 0.65, inflammation = 0.12)
  expect_type(params, "double")
  expect_true(length(params) > 0)
  expect_true(!is.null(names(params)))
})

test_that("cf_ctm_simulate returns data.frame", {
  skip_if_not_installed("reticulate")
  df <- cf_ctm_simulate(dose = 200, freq = 2, binding = 0.72, immune = 0.65,
                          inflammation = 0.12, horizon = 48)
  expect_s3_class(df, "data.frame")
  expect_true(nrow(df) > 0)
  expect_true("time_h" %in% names(df))
})

test_that("cf_rna_ctm_params handles modifications", {
  skip_if_not_installed("reticulate")
  params_psi <- cf_rna_ctm_params(modification = "pseudouridine")
  params_none <- cf_rna_ctm_params(modification = "none")
  expect_type(params_psi, "double")
  expect_type(params_none, "double")
  # Pseudouridine should have different (generally longer) half-life params
  expect_false(identical(params_psi, params_none))
})

test_that("cf_rna_ctm_simulate returns data.frame", {
  skip_if_not_installed("reticulate")
  df <- cf_rna_ctm_simulate(dose = 5, freq = 1, modification = "pseudouridine",
                              horizon = 48)
  expect_s3_class(df, "data.frame")
  expect_true(nrow(df) > 0)
})

test_that("cf_reg_metrics returns named vector", {
  skip_if_not_installed("reticulate")
  metrics <- cf_reg_metrics(c(1, 2, 3), c(1.1, 2.0, 2.9))
  expect_type(metrics, "double")
  expect_true("mae" %in% names(metrics) || "MAE" %in% toupper(names(metrics)))
})