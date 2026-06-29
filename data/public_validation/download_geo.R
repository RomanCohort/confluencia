
# R script: download_geo_data.R
if (!requireNamespace("GEOquery", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  BiocManager::install("GEOquery")
}

library(GEOquery)

# Download GSE109528
gse <- getGEO("GSE109528", GSEMatrix = TRUE, getGPL = FALSE)

# Save expression data
expr_data <- exprs(gse[[1]])
write.csv(expr_data, "D:/IGEM集成方案/data/public_validation/wesselhoeft_2018/GSE109528_expression.csv")

# Save sample info
sample_info <- pData(gse[[1]])
write.csv(sample_info, "D:/IGEM集成方案/data/public_validation/wesselhoeft_2018/GSE109528_samples.csv")

print("GSE109528 downloaded successfully!")
