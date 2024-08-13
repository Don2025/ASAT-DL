#!/usr/bin/env bash
#
# Exit the shell script immediately if any of the subsequent commands fails.
# immediately
set -e
#
################################################################################
# This script initializes Defects4J. In particular, it downloads and sets up:
# - the project's version control repositories
# - the Major mutation framework
# - the supported test generation tools
# - the supported code coverage tools (TODO)
################################################################################

# Print an error message and terminate the script.
# Takes one argument, a custom error message.
# Prints the supplied error message and a script termination notice to stderr.
# Terminates the script with exit code 1.
print_error_and_exit() {
  echo -e "${1} \nTerminating initialization... " >&2
  exit 1
}

# Get time of last data modification of a file
get_modification_timestamp() {
    local USAGE="Usage: get_modification_timestamp <file>"
    if [ "$#" != 1 ]; then
        print_error_and_exit "$USAGE"
    fi

    local f="$1"

    # The BSD version of stat does not support --version or -c
    if stat --version &> /dev/null; then
        # GNU version
        cmd="stat -c %Y $f"
    else
        # BSD version
        cmd="stat -f %m $f"
    fi

    local ts=$($cmd)
    echo "$ts"
}

# Check whether unzip is available
if ! unzip -v > /dev/null 2>&1; then
    print_error_and_exit "Couldn't find unzip to extract dependencies. Please install unzip and re-run this script."
fi

# Directories for project repositories and external libraries
MANUAL_DOWNLOAD="$(cd "$(dirname "$0")"; pwd)"
BASE="$(cd $MANUAL_DOWNLOAD; cd ..; pwd)"
DIR_REPOS="$BASE/project_repos"
DIR_LIB_GEN="$BASE/framework/lib/test_generation/generation"
DIR_LIB_RT="$BASE/framework/lib/test_generation/runtime"
DIR_LIB_GRADLE="$BASE/framework/lib/build_systems/gradle"

mkdir -p "$DIR_LIB_GEN" && mkdir -p "$DIR_LIB_RT" && mkdir -p "$DIR_LIB_GRADLE"


# Download project repositories if necessary
echo
echo "Setting up project repositories ... "
clean() {
    rm -rf \
    closure-compiler.git \
    commons-cli.git \
    commons-codec.git \
    commons-collections.git \
    commons-compress.git \
    commons-csv.git \
    commons-jxpath.git \
    commons-lang.git \
    commons-math.git \
    gson.git \
    jackson-core.git \
    jackson-databind.git \
    jackson-dataformat-xml \
    jfreechart \
    joda-time.git \
    jsoup.git \
    mockito.git \
    README 
}
cd $DIR_REPOS
ARCHIVE="defects4j-repos.zip"
# You have to wget "https://defects4j.org/downloads/defects4j-repos.zip",
# because File manual_download/defects4j-repos.zip is 418.97 MB; 
# this exceeds GitHub's file size limit of 100.00 MB.
cp "$MANUAL_DOWNLOAD/$ARCHIVE" $DIR_REPOS
clean()
unzip -q -u $ARCHIVE && mv defects4j/project_repos/* . && rm -r defects4j


# Download Major
echo
echo "Setting up Major ... "
MAJOR_ZIP="major-1.3.4_jre7.zip"
cp "$MANUAL_DOWNLOAD/$MAJOR_ZIP" $BASE
cd "$BASE" && unzip -q -u "$MAJOR_ZIP" \
           && rm "$MAJOR_ZIP" \
           && cp major/bin/.ant major/bin/ant

# Download EvoSuite
echo
echo "Setting up EvoSuite ... "
EVOSUITE_JAR="evosuite-1.1.0.jar"
EVOSUITE_RT_JAR="evosuite-standalone-runtime-1.1.0.jar"
cp "$MANUAL_DOWNLOAD/$EVOSUITE_JAR" $DIR_LIB_GEN
cp "$MANUAL_DOWNLOAD/$EVOSUITE_RT_JAR" $DIR_LIB_RT
# Set symlinks for the supported version of EvoSuite
(cd "$DIR_LIB_GEN" && ln -sf "$EVOSUITE_JAR" "evosuite-current.jar")
(cd "$DIR_LIB_RT" && ln -sf "$EVOSUITE_RT_JAR" "evosuite-rt.jar")

# Download Randoop
echo
echo "Setting up Randoop ... "
RANDOOP_VERSION="4.2.5"
RANDOOP_ZIP="randoop-${RANDOOP_VERSION}.zip"
RANDOOP_JAR="randoop-all-${RANDOOP_VERSION}.jar"
REPLACECALL_JAR="replacecall-${RANDOOP_VERSION}.jar"
COVEREDCLASS_JAR="covered-class-${RANDOOP_VERSION}.jar"
cp "$MANUAL_DOWNLOAD/$RANDOOP_ZIP" $DIR_LIB_GEN
(cd "$DIR_LIB_GEN" && unzip -q -u "$RANDOOP_ZIP")
# Set symlink for the supported version of Randoop
(cd "$DIR_LIB_GEN" && ln -sf "randoop-${RANDOOP_VERSION}/$RANDOOP_JAR" "randoop-current.jar")
(cd "$DIR_LIB_GEN" && ln -sf "randoop-${RANDOOP_VERSION}/$REPLACECALL_JAR" "replacecall-current.jar")
(cd "$DIR_LIB_GEN" && ln -sf "randoop-${RANDOOP_VERSION}/$COVEREDCLASS_JAR" "covered-class-current.jar")

# Download build system dependencies
echo
echo "Setting up Gradle dependencies ... "
cd "$DIR_LIB_GRADLE"

GRADLE_DISTS_ZIP="defects4j-gradle-dists.zip"
GRADLE_DEPS_ZIP="defects4j-gradle-deps.zip"

old_dists_ts=0
old_deps_ts=0

if [ -e $GRADLE_DISTS_ZIP ]; then
    old_dists_ts=$(get_modification_timestamp $GRADLE_DISTS_ZIP)
fi
if [ -e $GRADLE_DEPS_ZIP ]; then
    old_deps_ts=$(get_modification_timestamp $GRADLE_DEPS_ZIP)
fi

# Only download archive if the server has a newer file
cp "$MANUAL_DOWNLOAD/$GRADLE_DISTS_ZIP" DIR_LIB_GRADLE
cp "$MANUAL_DOWNLOAD/$GRADLE_DEPS_ZIP" $DIR_LIB_GRADLE
new_dists_ts=$(get_modification_timestamp $GRADLE_DISTS_ZIP)
new_deps_ts=$(get_modification_timestamp $GRADLE_DEPS_ZIP)

# Update gradle distributions/dependencies if a newer archive was available
[ "$old_dists_ts" != "$new_dists_ts" ] && mkdir "dists" && unzip -q -u $GRADLE_DISTS_ZIP -d "dists"
[ "$old_deps_ts" != "$new_deps_ts" ] && unzip -q -u $GRADLE_DEPS_ZIP

cd "$BASE"

# Download utility programs
echo
echo "Setting up utility programs ... "
BUILD_ANALYZER_VERSION="0.0.1"
BUILD_ANALYZER_JAR="build-analyzer-$BUILD_ANALYZER_VERSION.jar"
BUILD_ANALYZER_JAR_LOCAL="analyzer.jar"
cp "$MANUAL_DOWNLOAD/$BUILD_ANALYZER_JAR" "$BASE/framework/lib"
cd "$BASE/framework/lib"
rm -f "$BUILD_ANALYZER_JAR_LOCAL"
ln -s "$BUILD_ANALYZER_JAR" "$BUILD_ANALYZER_JAR_LOCAL"

echo
echo "Defects4J successfully initialized."