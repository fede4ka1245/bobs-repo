import uuid


def generate_unique_filename(original_filename):
    _, extension = original_filename.rsplit('.', 1)
    unique_filename = f"{uuid.uuid4()}.{extension}"
    return unique_filename