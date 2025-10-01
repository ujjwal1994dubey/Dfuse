```
gemini_data_formulator/
│
├── __init__.py
├── main.py                  # Entry point / CLI / API integration
│
├── core/
│   ├── gemini_client.py     # Run Gemini, simulate responses, direct analysis
│   ├── parser.py            # Parse structured/natural responses
│   ├── executor.py          # Execute pandas transformations safely
│   ├── schema.py            # Auto-detect dimensions/measures, JSON safety
│   ├── agent.py             # Pandas agent logic & validation
│   └── metrics.py           # Metric calculations & formatting
│
├── utils/
│   ├── charting.py          # Suggest chart types
│   ├── logging_utils.py     # Transformation logs, safe error handling
│   └── text_utils.py        # String matching, keyword detection
│
├── exploration/
│   ├── explorer.py          # explore_data_enhanced, explore_data_original
│   └── tabular_parser.py    # Convert Pandas/agent outputs to tables
│
└── tests/
    ├── test_parser.py
    ├── test_executor.py
    ├── test_agent.py
    └── test_metrics.py
