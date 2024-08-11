from pydantic import BaseModel
import vectorbt as vbt
from vectorbt.portfolio import Portfolio
from typing import Any, Optional

class BaseAnalysis(BaseModel):
    price_data: Any
    portfolio: Optional[Portfolio] = None