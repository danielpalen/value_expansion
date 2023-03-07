function setup {
  echo "Base Conda: $(which conda)"
  eval "$($(which conda) shell.bash hook)"
  conda activate value_expansion
  echo "Conda Env:  $(which conda)"

  export GTIMER_DISABLE='1'
  echo "GTIMER_DISABLE: $GTIMER_DISABLE"

  export WANDB_DIR='/tmp'

  cd $SCRIPT_PATH/../experiments
  echo "Working Directory:  $(pwd)"
  echo
}
