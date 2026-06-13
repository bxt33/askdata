import pandas as pd

from main import parse_column_value_change_intent
from storage import AppStorage
from utils import DataframeAgentFacade


def test_bar_pipeline_without_api_key():
    df = pd.read_csv("house_price.csv")
    result = DataframeAgentFacade("").analyze(df, "furnishingstatus 的分布柱状图")

    assert "bar" in result
    assert result["_executed_tools"] == ["value_counts", "plot_bar"]


def test_shape_answer_without_api_key():
    df = pd.read_csv("personal_data.csv")
    result = DataframeAgentFacade("").analyze(df, "数据有多少行多少列")

    assert "answer" in result
    assert "22" in result["answer"]


def test_csv_upload_is_persisted_as_sqlite_table(tmp_path):
    storage = AppStorage(tmp_path / "test.db", tmp_path / "data")
    dataset_id = storage.save_uploaded_dataset("small.csv", b"a,b\n1,x\n2,y\n")
    meta = storage.get_dataset(dataset_id)
    df = storage.read_dataset(dataset_id)

    assert meta is not None
    assert meta["table_name"].startswith("dataset_")
    assert meta["content_type"] == "table"
    assert df.shape == (2, 2)


def test_database_dataset_column_values_can_be_persisted(tmp_path):
    storage = AppStorage(tmp_path / "test.db", tmp_path / "data")
    dataset_id = storage.register_dataframe("managed", pd.DataFrame({"category": ["a", "b"], "amount": [1, 2]}))

    count = storage.update_dataset_column_values(dataset_id, "category", 1)
    df = storage.read_dataset(dataset_id)

    assert count == 2
    assert df["category"].tolist() == [1, 1]


def test_uploaded_dataset_column_values_are_not_mutable(tmp_path):
    storage = AppStorage(tmp_path / "test.db", tmp_path / "data")
    dataset_id = storage.save_uploaded_dataset("small.csv", b"category,amount\na,1\nb,2\n")

    try:
        storage.update_dataset_column_values(dataset_id, "category", 1)
    except PermissionError:
        pass
    else:
        raise AssertionError("uploaded datasets must not allow online value mutation")


def test_value_change_request_requires_admin_approval(tmp_path):
    storage = AppStorage(tmp_path / "test.db", tmp_path / "data")
    dataset_id = storage.register_dataframe("managed", pd.DataFrame({"category": ["a", "b"], "amount": [1, 2]}))

    request_id = storage.create_value_change_request(dataset_id, "category", 1, "analyst")
    before = storage.read_dataset(dataset_id)
    pending = storage.list_value_change_requests("pending")

    assert before["category"].tolist() == ["a", "b"]
    assert pending[0]["id"] == request_id
    assert pending[0]["new_value"] == 1

    count = storage.approve_value_change_request(request_id)
    after = storage.read_dataset(dataset_id)

    assert count == 2
    assert after["category"].tolist() == [1, 1]
    assert storage.list_value_change_requests("pending") == []


def test_column_value_change_intent_accepts_common_chinese_forms():
    columns = ["date", "category", "amount"]

    assert parse_column_value_change_intent("把category字段全部改为1", columns) == ("category", 1)
    assert parse_column_value_change_intent("将category字段的所有值替换为1", columns) == ("category", 1)
    assert parse_column_value_change_intent("category字段的所有值设置为测试", columns) == ("category", "测试")
