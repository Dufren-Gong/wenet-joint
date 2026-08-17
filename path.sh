export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

environment_name=cwenet-swbd
active_environment_shell_path=./activate_python.sh
if [ -f ${active_environment_shell_path} ]; then
    . ${active_environment_shell_path} ${environment_name}
    echo "activated miniconda environment $environment_name."
else
    echo "you can create a ${active_environment_shell_path} to auto activate environment."
fi
