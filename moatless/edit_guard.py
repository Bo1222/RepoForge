"""Optimistic edit tokens include both the overlay and its underlying source."""

from moatless.message_history.policy import content_hash


class EditConflict(ValueError):
    pass


def source_version(repository, path, file=None):
    disk = repository.get_file_content(path)
    overlay = file.content if file is not None else disk
    if file is not None and file._is_new and not file.was_edited and disk is None:
        overlay = None
    if not repository.shadow_mode and file is not None and disk != overlay:
        raise EditConflict("Cached source differs from the current file; read it again before editing")
    return content_hash({"path": path, "overlay": overlay, "source": disk})
