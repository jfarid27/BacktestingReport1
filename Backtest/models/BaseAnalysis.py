from vectorbt.portfolio import Portfolio
from typing import Any, Optional

class BaseAnalysis():
    price_data: Any
    portfolio: Optional[Portfolio] = None
    pass