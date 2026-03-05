"""Shared test fixtures for the AI Strategic Partner test suite."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from picp.vector_clock import VectorClock
from picp.message import PICPContext, PICPEvent


@pytest.fixture
def pillar_names() -> list[str]:
    """Standard pillar names used throughout the system."""
    return ["knowledge", "reasoning", "simulation", "trust"]


@pytest.fixture
def vector_clock(pillar_names: list[str]) -> VectorClock:
    """A fresh VectorClock initialised with all four pillars at zero."""
    return VectorClock(pillars=pillar_names)


@pytest.fixture
def picp_context() -> PICPContext:
    """A fresh PICPContext for testing pipeline flows."""
    return PICPContext.new(query="What is the projected cashflow for Q1 2025?")
