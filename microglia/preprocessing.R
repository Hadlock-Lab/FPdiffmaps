library(reticulate)
use_condaenv("oldReliable") #replace with your local conda environment or python location
py_config()

library(Seurat)
library(SeuratDisk)
library(anndata)

data <- read_h5ad("/Users/andrew/Documents/ROSMAP/velocityOut/patelData/patelOlahCountWithOriginalClusters.h5ad")
data <- CreateSeuratObject(counts = t(data$X), meta.data = data$obs, min.cells=5, min.features=400)

data[["percent.mt"]] = PercentageFeatureSet(data, pattern = "^MT-")

#follow Patel's quality control pipeline
data = subset(data, subset = nFeature_RNA < 8000 & nCount_RNA >500 & nCount_RNA <46425 & percent.mt <10)
#data = NormalizeData(data)
#data = FindVariableFeatures(data, nfeatures=2000)
data.list = SplitObject(data, split.by="batch")
data.list <- lapply(X = data.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

data.anchors=FindIntegrationAnchors(object.list = data.list, anchor.features = 2000)
data.combined = IntegrateData(anchorset = data.anchors)

SaveH5Seurat(data.combined, filename = "/Users/andrew/Documents/ROSMAP/velocityOut/patelData/patelOlahDataProcessedSeurat_11_16_22.h5seurat")
Convert("/Users/andrew/Documents/ROSMAP/velocityOut/patelData/patelOlahDataProcessedSeurat_11_16_22.h5seurat", dest = "h5ad") 
#only do previous line if you want to move back to python. otherwise, keep in h5seurat format

