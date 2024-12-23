from typing import Union
from pathlib import Path
import pyarrow as pa
import os


class RetrievalDatabaseMetadataProvider:
    """Metadata provider for the retrieval database.

    Args:
        metadata_dir (str): Path to the metadata directory.
    """

    def __init__(self, metadata_dir: str, getitem_only_caption: bool = False):
        metadatas = [str(a) for a in sorted(Path(metadata_dir).glob("**/*")) if a.is_file() and os.path.splitext(a)[-1] == '.arrow']
        self.table = pa.concat_tables(
            [
                pa.ipc.RecordBatchFileReader(pa.memory_map(metadata, "r")).read_all()
                for metadata in metadatas
            ]
        )

        self.getitem_only_caption = getitem_only_caption

    def __len__(self):
        return len(self.table)

    def get(self, ids):
        """Get the metadata for the given ids.

        Args:
            ids (list): List of ids.
        """

        columns = self.table.schema.names
        end_ids = [i + 1 for i in ids]
        t = pa.concat_tables([self.table[start:end] for start, end in zip(ids, end_ids)])
        return t.select(columns).to_pandas().to_dict("records")
    
    def __getitem__(self, key: Union[slice, int, list]):
        if isinstance(key, int):
            ids = [key]
        elif isinstance(key, list):
            ids = key
        else:
            ids = list(range(key.start if key.start else 0, key.stop if key.stop else len(self), key.step if key.step else 1))
        
        result = self.get(ids)
        if not self.getitem_only_caption:
            return result
        
        if isinstance(key, int):
            return result[0]['caption']
        else:
            return [x['caption'] for x in result]
