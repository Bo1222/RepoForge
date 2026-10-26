#!/bin/bash
set -e

# Default values
BRANCH="main"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "🔄 Updating repoforge repository to latest from branch '$BRANCH'"

# Assume we're in the container at /opt/repoforge
cd /opt/repoforge

# Update the repository
echo "📂 Fetching latest changes..."
git fetch origin
git checkout $BRANCH
git pull origin $BRANCH

# Update dependencies
echo "📦 Updating dependencies..."
uv sync --frozen --compile-bytecode --all-extras

echo "✅ Update complete! Repository is now at: $(git rev-parse --short HEAD)" 