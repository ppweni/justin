"""Justin (inference): a time series and questions in, answers out.

Timestral or Chronos-2 reads the signal, MiniLM the question and the answer options, and a small trained
cross-attention head picks an answer; a location head points at where something happens.
"""

from .pretrained import Answer, Found, Justin, Location, Question, load_location, load_pretrained

__all__ = ["Answer", "Found", "Justin", "Location", "Question", "load_location", "load_pretrained"]
