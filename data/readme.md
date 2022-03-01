# Data 

## Metadata

numerai_signals_metadata_public_2021.csv: Metadata of US stocks (2003-2021) within Numerai Signals universe. 
It is prepared by mapping CRSP and Compustat historical records with Numerai.

  - bloomberg_ticker:
  - numerai_exit_date: Last Date in Numerai Universe 
  - numerai_entry_date: First Date in Numerai Universe 
  - hconm: Company Name
  - hcik: SEC EDGAR cik
  - map_start: First valid date of this mapping record
  - map_end: Last valid date of this mapping record

Note: For some bloomberg tickers there are multiple ciks that mapped. This is due to corporate actions and some inconsistenecy between databases. 
These tickers are removed in the downstream data preparation process.

