import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataHandler import DataHandler
from datetime import datetime
import pytest

def test_download_data():
    handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    df = handler.download_data()
    assert not df.empty, "Downloaded data should not be empty."

def test_save_and_load_csv(tmp_path):
    handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    df = handler.download_data()
    
    filepath = tmp_path / "test_data.csv"
    handler.save_to_csv(filepath)
    
    new_handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    new_handler.load_from_csv(filepath)
    new_df = new_handler.data
    
    assert df.equals(new_df), "Loaded data should match saved data."

def test_get_backtrader_data_without_download():
    handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    with pytest.raises(ValueError, match="Data not downloaded yet."):
        handler.get_backtrader_data()

def test_save_without_data(tmp_path):
    handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    filepath = tmp_path / "empty.csv"
    with pytest.raises(ValueError, match="No data to save."):
        handler.save_to_csv(filepath)

def test_load_invalid_file(tmp_path):
    handler = DataHandler("QQQ", "2021-01-01", "2021-01-10")
    filepath = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        handler.load_from_csv(filepath)