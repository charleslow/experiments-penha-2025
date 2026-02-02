#!/bin/bash
# Block dangerous commands even when using --dangerously-skip-permissions

# Read stdin (the JSON input)
INPUT=$(cat)

# Extract command using grep/sed (no jq dependency)
COMMAND=$(echo "$INPUT" | grep -o '"command":"[^"]*"' | sed 's/"command":"//;s/"$//')

# Patterns to block
BLOCKED=(
  "rm -rf"
  "rm -r "
  "git clean"
  "git reset --hard"
  "git push --force"
  "git push -f"
  "sudo "
  "chmod 777"
  "> /dev/"
  "mkfs"
  "dd if="
)

for pattern in "${BLOCKED[@]}"; do
  if [[ "$COMMAND" == *"$pattern"* ]]; then
    echo "BLOCKED: Command contains dangerous pattern '$pattern'" >&2
    exit 2  # exit 2 = block the command
  fi
done

exit 0  # allow the command
