"""
Universal Cloud Storage & Weights Synchronization Manager.
Supports:
- Google Cloud Storage (GCS)
- Amazon Web Services (AWS S3)
- Hugging Face Hub / Local disk fallback
"""

import os
import shutil

class CloudStorageManager:
    """
    Manages cloud synchronization of model weights (yolov8n.pt, cnn_lstm_traffic.pt)
    and traffic data files across AWS S3, Google Cloud Storage, or Local storage.
    """
    def __init__(self):
        self.provider = os.environ.get("CLOUD_STORAGE_PROVIDER", "local").lower()
        self.bucket_name = os.environ.get("CLOUD_STORAGE_BUCKET", "emergency-routing-weights")
        self._init_client()

    def _init_client(self):
        self.client = None
        if self.provider == "gcs":
            try:
                from google.cloud import storage
                self.client = storage.Client()
                print(f"[Cloud Storage] Connected to Google Cloud Storage (Bucket: {self.bucket_name})")
            except Exception as e:
                print(f"[Cloud Storage Warning] GCS client initialization failed: {e}. Falling back to local storage.")
                self.provider = "local"
        elif self.provider in ("s3", "aws"):
            try:
                import boto3
                self.client = boto3.client("s3")
                print(f"[Cloud Storage] Connected to Amazon S3 (Bucket: {self.bucket_name})")
            except Exception as e:
                print(f"[Cloud Storage Warning] AWS S3 client initialization failed: {e}. Falling back to local storage.")
                self.provider = "local"
        else:
            print("[Cloud Storage] Operating in local/container storage mode (zero cloud credentials needed).")

    def download_file(self, remote_key: str, local_path: str) -> bool:
        """Downloads a file from cloud storage to local disk."""
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return True

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if self.provider == "gcs" and self.client:
            try:
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.download_to_filename(local_path)
                print(f"[Cloud Storage] Downloaded {remote_key} from GCS -> {local_path}")
                return True
            except Exception as e:
                print(f"[Cloud Storage Error] Failed to download {remote_key} from GCS: {e}")
                return False

        elif self.provider in ("s3", "aws") and self.client:
            try:
                self.client.download_file(self.bucket_name, remote_key, local_path)
                print(f"[Cloud Storage] Downloaded {remote_key} from S3 -> {local_path}")
                return True
            except Exception as e:
                print(f"[Cloud Storage Error] Failed to download {remote_key} from S3: {e}")
                return False

        return False

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        """Uploads a local file to cloud storage bucket."""
        if not os.path.exists(local_path):
            print(f"[Cloud Storage Warning] Cannot upload nonexistent file: {local_path}")
            return False

        if self.provider == "gcs" and self.client:
            try:
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.upload_from_filename(local_path)
                print(f"[Cloud Storage] Uploaded {local_path} -> GCS ({self.bucket_name}/{remote_key})")
                return True
            except Exception as e:
                print(f"[Cloud Storage Error] GCS upload failed: {e}")
                return False

        elif self.provider in ("s3", "aws") and self.client:
            try:
                self.client.upload_file(local_path, self.bucket_name, remote_key)
                print(f"[Cloud Storage] Uploaded {local_path} -> S3 ({self.bucket_name}/{remote_key})")
                return True
            except Exception as e:
                print(f"[Cloud Storage Error] S3 upload failed: {e}")
                return False

        return False

    def get_status(self):
        """Returns cloud storage provider health and connection metadata."""
        return {
            "provider": self.provider,
            "bucket": self.bucket_name if self.provider != "local" else None,
            "connected": self.client is not None or self.provider == "local"
        }

# Global singleton
storage_manager = CloudStorageManager()
