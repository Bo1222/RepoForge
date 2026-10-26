"""Actual Git directory entries must be recoverable without following external links."""
import subprocess
from types import SimpleNamespace

from repoforge.flow.durable import workspace_fingerprint


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT)


def init(repo):
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'Fingerprint test')
    git(repo, 'config', 'user.email', 'test@example.invalid')


def fingerprint(repo, checkpoint):
    workspace = SimpleNamespace(repository=SimpleNamespace(repo_path=str(repo), shadow_mode=True),
                                runtime=None, _environment=None)
    return workspace_fingerprint(workspace, str(checkpoint))['repository']['content_hash']


def test_gitlink_tracks_worktree_and_commit_without_hashing_git_metadata(tmp_path):
    repo = tmp_path / 'repo'
    init(repo)
    nested = repo / 'helpers'
    init(nested)
    (nested / 'public.py').write_text('value = 1\n')
    git(nested, 'add', '.')
    git(nested, 'commit', '-qm', 'public dependency')
    git(repo, 'add', 'helpers')
    git(repo, 'commit', '-qm', 'gitlink')
    assert git(repo, 'ls-files', '--stage').startswith(b'160000 ')
    checkpoint = nested / 'state.sqlite'
    original = fingerprint(repo, checkpoint)
    checkpoint.write_text('state')
    checkpoint.with_suffix('.sqlite-wal').write_text('wal')
    assert fingerprint(repo, checkpoint) == original
    (nested / 'public.py').write_text('value = 2\n')
    assert fingerprint(repo, checkpoint) != original
    (nested / 'public.py').write_text('value = 1\n')
    assert fingerprint(repo, checkpoint) == original
    git(nested, 'commit', '--allow-empty', '-qm', 'different dependency commit')
    assert fingerprint(repo, checkpoint) != original


def test_symlink_target_changes_are_tracked_without_reading_external_contents(tmp_path):
    repo = tmp_path / 'repo'
    init(repo)
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'file'
    secret.write_text('outside workspace')
    link = repo / 'link'
    link.symlink_to(outside, target_is_directory=True)
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'link')
    original = fingerprint(repo, repo / 'state.sqlite')
    secret.write_text('external contents changed')
    assert fingerprint(repo, repo / 'state.sqlite') == original
    link.unlink()
    link.symlink_to(outside / 'missing')
    assert fingerprint(repo, repo / 'state.sqlite') != original
    link.unlink()
    link.symlink_to('link')
    cyclic = fingerprint(repo, repo / 'state.sqlite')
    assert cyclic != original
    assert fingerprint(repo, repo / 'state.sqlite') == cyclic
