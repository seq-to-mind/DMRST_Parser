* Here are some details of the results on English RST corpora (`RST-DT` and `GUM`):
1. In our experimental setting, we tested our multilingual parser on `English RST-DT` and `GUM` separately.
2. The `EN-DT` scores reported in our DMRST paper is only on the `English RST-DT` test test, as it is usually used as the benchmark parsing corpus.
 
* Additionally, here is the Micro F1 result on the `GUM` corpus in our experiment:

| Span  | Nuclearity | Relation | Segmentation Accuracy | EDU Label                  | Metrics                                |
|-------|------------|----------|-----------------------|----------------------------|----------------------------------------|
| 84.96 | 71.69      | 51.48    | 1.00                  | Gold EDU Segmentation      | RST Parseval (Marcu, 2000)             |
| 75.05 | 61.15      | 42.89    | 0.921                 | Predicted EDU Segmentation | RST Parseval (Marcu, 2000)             |
| 69.93 | 54.51      | 42.06    | 1.00                  | Gold EDU Segmentation      | Original Parseval (Morey et al., 2017) |
| 58.02 | 43.94      | 34.16    | 0.921                 | Predicted EDU Segmentation | Original Parseval (Morey et al., 2017) |

