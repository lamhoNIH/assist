# apt-get install procps to install top and monitor mem usage
# tried 20G and the process got killed

date
docker run -m32g --rm -e archive_path="module_extraction/run1" -v "/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data":/assist/Data assist/module_extraction:0.1.0
date