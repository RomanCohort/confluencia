library(confluencia)

# Set Python path
cf_use_python("C:/Program Files/Python313/python.exe")

# Test 1: CTM params
cat("=== Test 1: CTM params ===\n")
params <- cf_ctm_params(binding=0.72, immune=0.65, inflammation=0.12)
print(params)

# Test 2: CTM simulate
cat("=== Test 2: CTM simulate ===\n")
pk <- cf_ctm_simulate(dose=200, freq=2, binding=0.72, horizon=72)
cat("nrow=", nrow(pk), "\n")
cat("cols=", paste(names(pk), collapse=", "), "\n")
cat("max protein=", max(pk$translated_protein), "\n")

# Test 3: RNA-CTM params
cat("=== Test 3: RNA-CTM params ===\n")
rna_params <- cf_rna_ctm_params(modification="pseudouridine")
print(rna_params)

# Test 4: RNA-CTM simulate
cat("=== Test 4: RNA-CTM simulate ===\n")
rna_pk <- cf_rna_ctm_simulate(dose=5, freq=1, modification="pseudouridine", horizon=168)
cat("nrow=", nrow(rna_pk), "\n")
cat("max protein=", max(rna_pk$translated_protein), "\n")

# Test 5: MHC encode
cat("=== Test 5: MHC encode ===\n")
enc <- cf_mhc_encode("SLYNTVATL", "HLA-A*02:01")
cat("MHC-I dim=", length(enc), "\n")

# Test 6: MHC detect class
cat("=== Test 6: MHC detect class ===\n")
cls <- cf_mhc_detect_class("HLA-A*02:01")
cat("Class=", cls, "\n")

# Test 7: Reg metrics
cat("=== Test 7: Reg metrics ===\n")
m <- cf_reg_metrics(c(1,2,3), c(1.1,2,2.9))
print(m)

# Test 8: Immunogenicity
cat("=== Test 8: Immunogenicity ===\n")
imm <- cf_circrna_immunogenicity("ACGUACGUACGUACGU")
print(imm)

# Test 9: List plugins
cat("=== Test 9: List plugins ===\n")
pl <- cf_list_plugins()
print(pl)

cat("=== ALL TESTS PASSED ===\n")