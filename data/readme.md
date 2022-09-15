# Data 

## Metadata

numerai_signals_stock_metadata_public_2021.csv: Metadata of US stocks (2003-2021) within Numerai Signals universe. 
It is prepared by mapping CRSP and Compustat historical records with Numerai.

  - bloomberg_ticker:
  - numerai_exit_date: Last Date in Numerai Universe 
  - numerai_entry_date: First Date in Numerai Universe 
  - hconm: Company Name
  - hcik: SEC EDGAR cik
  - crsp_start: First valid date of the ticker mapped to CRSP
  - crsp_end: Last valid date of the ticker mapped to CRSP

Note: For some bloomberg tickers there are multiple ciks that mapped. This is due to corporate actions and some inconsistenecy between databases. 
These tickers are removed in the downstream data preparation process.


## Dataset 
    
- Dataset version 4: Price, Financials and Sentiment for US stocks (2003-2021)
  - Price Features: Basic Statistics, Signature and Catch22 features using CRSP Stock Database
  - Financials Features: 204 features from [Open Source Asset Pricing](https://www.openassetpricing.com/data/)
  - Sentiment Features: Sentiment stocks from Ravenpack 
  - Link: [https://zenodo.org/record/6524559](https://zenodo.org/record/6524559)
    

