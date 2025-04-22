#!/bin/bash

# Color definitions for logging
export ERROR="\e[31m"
export WARN="\e[33m"
export SUCCESS="\e[32m"
export INFO="\e[36m"
export RESET_TEXT_COLOR="\e[0m"
export NEW_LINE="\n"

# Github repositories
export FRONTEND="https://github.com/pib-rocks/cerebra.git"
export BACKEND="https://github.com/pib-rocks/pib-backend.git"
export APP_DIR="$HOME/app"
export BACKEND_DIR="$APP_DIR/pib-backend"
export FRONTEND_DIR="$APP_DIR/cerebra"
export SETUP_INSTALLATION_DIR="$BACKEND_DIR/setup/installation_scripts"

# Installation status tracking
export STATUS_DIR="$HOME/.pib_setup_status"
export MAX_RETRIES=3

# Function to support printing consistent log messages
function print() {
    local color=$1
    local text=$2

    # If only one argument is provided, assume it is the text
    if [ -z "$text" ]; then
        text=$color
        color="RESET_TEXT_COLOR"
    fi

    # Check if the provided color exists
    if [ -n "$color" ] && [ -z "${!color}" ]; then
        color="RESET_TEXT_COLOR"
    fi

    # Print the text in the specified color
    echo -e "${!color}[$(date -u)][[ ${text} ]]${RESET_TEXT_COLOR}"
}

function command_exists() {
    command -v "$@" >/dev/null 2>&1
}

# Get Linux distribution name, e.g. 'ubuntu', 'debian'
get_distribution() {
    local distribution=""
    if [ -r /etc/os-release ]; then
        distribution="$(. /etc/os-release && echo "$ID")"
    fi
    echo "$distribution"
}

# Get Linux distribution version, e.g. (ubuntu) 'jammy', (debian) 'bookworm'
get_dist_version() {
  local distribution=$1
  case "$distribution" in

    ubuntu)
        if command_exists lsb_release; then
            dist_version="$(lsb_release --codename | cut -f2)"
        fi
        if [ -z "$dist_version" ] && [ -r /etc/lsb-release ]; then
            dist_version="$(. /etc/lsb-release && echo "$DISTRIB_CODENAME")"
        fi
        ;;

    debian | raspbian)
        dist_version="$(sed 's/\/.*//' /etc/debian_version | sed 's/\..*//')"
        case "$dist_version" in
        12)
            dist_version="bookworm"
            ;;
        11)
            dist_version="bullseye"
            ;;
        10)
            dist_version="buster"
            ;;
        esac
        ;;
    esac
    echo "$dist_version" |  tr '[:upper:]' '[:lower:]'
}

# Function to check if a step has been completed successfully
function is_step_completed() {
    local step_name=$1
    [ -f "$STATUS_DIR/${step_name}.completed" ]
}

# Function to mark a step as completed
function mark_step_completed() {
    local step_name=$1
    mkdir -p "$STATUS_DIR"
    touch "$STATUS_DIR/${step_name}.completed"
    print SUCCESS "Marked step '${step_name}' as completed"
}

# Function to retry a step with exponential backoff
function retry_step() {
    local step_name=$1
    local step_function=$2
    local retry_count=0
    local wait_time=5
    
    if is_step_completed "$step_name"; then
        print INFO "Step '${step_name}' was already completed successfully, skipping"
        return 0
    fi
    
    print INFO "Starting step: ${step_name}"
    
    until [ $retry_count -ge $MAX_RETRIES ]; do
        if [ $retry_count -gt 0 ]; then
            print WARN "Retrying '${step_name}' (attempt ${retry_count}/${MAX_RETRIES})"
            sleep $wait_time
            # Increase wait time for next retry (exponential backoff)
            wait_time=$((wait_time * 2))
        fi
        
        if $step_function; then
            mark_step_completed "$step_name"
            return 0
        else
            print ERROR "Step '${step_name}' failed"
            ((retry_count++))
        fi
    done
    
    print ERROR "Step '${step_name}' failed after ${MAX_RETRIES} attempts"
    return 1
}

function remove_apps_impl() {
    print INFO "Removing unused default software"

    if ! [ "$DISTRIBUTION" == "ubuntu" ]; then
        print INFO "Not using Ubuntu 22.04; skipping removing unused default software"
        return 0
    fi

    PACKAGES_TO_BE_REMOVED=("aisleriot" "gnome-sudoku" "ace-of-penguins" "gbrainy" "gnome-mines" "gnome-mahjongg" "libreoffice*" "thunderbird*")
    installed_packages_to_be_removed=""

    # Create a list of all currently installed packaged that should be removed to reduce software bloat
    for package_name in "${PACKAGES_TO_BE_REMOVED[@]}"; do
        if dpkg-query -W -f='${Status}\n' "$package_name" 2>/dev/null | grep -q "install ok installed"; then
            installed_packages_to_be_removed+="$package_name "
        fi
    done

    # Remove unnecessary packages, if any are found
    if [ -n "$installed_packages_to_be_removed" ]; then
        sudo apt-get -y purge $installed_packages_to_be_removed && \
        sudo apt-get autoclean || return 1
    fi

    print SUCCESS "Removed unused default software"
    return 0
}

function install_system_packages_impl() {
    print INFO "Installing system packages"
    sudo apt update -qq && \
    sudo apt-get install -y git curl openssh-server >/dev/null
    if [ $? -ne 0 ]; then
        return 1
    fi
    print SUCCESS "Installing system packages completed"
    return 0
}

# Function to clone pib repositories to APP_DIR (~/app) directory
function clone_repositories_impl() {
    # Validate branches
    if ! command_exists git; then
        print ERROR "git not found"
        return 1
    fi

    if ! git ls-remote --exit-code --heads "$FRONTEND" "$BRANCH_FRONTEND" >/dev/null 2>&1; then
        print ERROR "Branch '${BRANCH_FRONTEND}' for Cerebra not found"
        return 1
    fi
    if ! git ls-remote --exit-code --heads "$BACKEND" "$BRANCH_BACKEND" >/dev/null 2>&1; then
        print ERROR "Branch '${BRANCH_BACKEND}' for pib-backend not found"
        return 1
    fi

    print INFO "Using branch '${BRANCH_FRONTEND}' for Cerebra, '${BRANCH_BACKEND}' for pib-backend"

    # Clone Repositories
    if [ ! -d "$APP_DIR" ]; then
        mkdir $APP_DIR || return 1
        print INFO "${APP_DIR} created"
    fi

    # Clone or update backend repository
    if [ -d "$BACKEND_DIR" ]; then
        print INFO "pib-backend repository exists, updating..."
        (cd "$BACKEND_DIR" && git fetch && git checkout "$BRANCH_BACKEND" && git pull) || return 1
    else
        git clone --recurse-submodules -b "$BRANCH_BACKEND" $BACKEND "$BACKEND_DIR" || return 1
    fi

    # Clone or update frontend repository
    if [ -d "$FRONTEND_DIR" ]; then
        print INFO "cerebra repository exists, updating..."
        (cd "$FRONTEND_DIR" && git fetch && git checkout "$BRANCH_FRONTEND" && git pull) || return 1
    else
        git clone --recurse-submodules -b "$BRANCH_FRONTEND" $FRONTEND "$FRONTEND_DIR" || return 1
    fi

    print SUCCESS "Completed cloning repositories to $APP_DIR"
    return 0
}

# Install update script; move animated eyes, etc.
function move_setup_files_impl() {
    local update_target_dir="/usr/local/bin"
    sudo cp "$BACKEND_DIR/setup/update-pib.sh" "$update_target_dir/update-pib" || return 1
    sudo chmod 755 "$update_target_dir/update-pib" || return 1
    print SUCCESS "Installed update script"

    cp "$BACKEND_DIR/setup/setup_files/pib-eyes-animated.gif" "$HOME/Desktop/pib-eyes-animated.gif" || return 1
    print SUCCESS "Moved animated eyes to Desktop"

    # Add HTML that opens Cerebra + Database to the Desktop
    printf '<meta content="0; url=http://localhost" http-equiv=refresh>' > "$HOME/Desktop/Cerebra.html" || return 1
    printf '<meta content="0; url=http://localhost:8000" http-equiv=refresh>' > "$HOME/Desktop/pib_data.html" || return 1
    
    return 0
}

function install_DBbrowser_impl() {
    sudo apt update -qq && \
    sudo apt install -y sqlitebrowser || return 1
    print SUCCESS "Installed DB browser"
    return 0
}

function install_BrickV_impl() {
    wget https://download.tinkerforge.com/apt/$(. /etc/os-release; echo $ID)/tinkerforge.asc -q -O - | sudo tee /etc/apt/trusted.gpg.d/tinkerforge.asc > /dev/null || return 1
    echo "deb https://download.tinkerforge.com/apt/$(. /etc/os-release; echo $ID $VERSION_CODENAME) main" | sudo tee /etc/apt/sources.list.d/tinkerforge.list || return 1
    sudo apt update -qq || return 1
    sudo apt install -y brickv || return 1
    sudo apt install -y python3-tinkerforge || return 1 #python API Bindings
    print SUCCESS "Installed brick viewer and python API bindings"
    return 0
}

function disable_power_notification_impl() {
    local file="/boot/firmware/config.txt"
    
    if [ -f "$file" ]; then
        echo "Disabling under-voltage warnings..."
        grep -q "avoid_warnings=2" "$file" || echo "avoid_warnings=2" | sudo tee -a "$file" > /dev/null || return 1

        echo "Preventing CPU throttling..."
        grep -q "force_turbo=1" "$file" || echo "force_turbo=1" | sudo tee -a "$file" > /dev/null || return 1
    fi

    echo "Installing and configuring watchdog service..."
    sudo apt-get install -y watchdog || return 1
    sudo systemctl enable watchdog || return 1
    sudo systemctl start watchdog || return 1

    echo "Modifying watchdog configuration..."
    sudo sed -i 's/#reboot=1/reboot=0/' /etc/watchdog.conf || return 1

    echo "Disabling kernel panic reboots..."
    grep -q "kernel.panic = 0" /etc/sysctl.conf || echo "kernel.panic = 0" | sudo tee -a /etc/sysctl.conf || return 1

    sudo sysctl -p || return 1
    return 0
}

function set_system_settings_impl() {
    if [ -f "$SETUP_INSTALLATION_DIR/set_system_settings.sh" ]; then
        source "$SETUP_INSTALLATION_DIR/set_system_settings.sh" || return 1
        return 0
    else
        print WARN "System settings script not found, skipping"
        return 0
    fi
}

function install_docker_impl() {
    if [ -f "$SETUP_INSTALLATION_DIR/docker_install.sh" ]; then
        source "$SETUP_INSTALLATION_DIR/docker_install.sh" || return 1
        return 0
    else
        print ERROR "Docker installation script not found"
        return 1
    fi
}

function install_local_impl() {
    if [ -f "$SETUP_INSTALLATION_DIR/local_install.sh" ]; then
        source "$SETUP_INSTALLATION_DIR/local_install.sh" || return 1
        return 0
    else
        print ERROR "Local installation script not found"
        return 1
    fi
}

# Wrapper functions for retry logic
function remove_apps() {
    retry_step "remove_apps" remove_apps_impl
}

function install_system_packages() {
    retry_step "install_system_packages" install_system_packages_impl
}

function clone_repositories() {
    retry_step "clone_repositories" clone_repositories_impl
}

function move_setup_files() {
    retry_step "move_setup_files" move_setup_files_impl
}

function install_DBbrowser() {
    retry_step "install_DBbrowser" install_DBbrowser_impl
}

function install_BrickV() {
    retry_step "install_BrickV" install_BrickV_impl
}

function disable_power_notification() {
    retry_step "disable_power_notification" disable_power_notification_impl
}

function set_system_settings() {
    retry_step "set_system_settings" set_system_settings_impl
}

function install_docker() {
    retry_step "install_docker" install_docker_impl
}

function install_local() {
    retry_step "install_local" install_local_impl
}

# Clean setup files if local install + remove user from sudoers file again
function cleanup() {
    if [ "$INSTALL_METHOD" = "legacy" ]; then
        sudo rm -r "$HOME/app"
        print INFO "Removed repositories from $HOME due to local installation"
    fi
    sudo rm /etc/sudoers.d/"$USER"
    
    # Don't track this step as we always want to run it
}

function show_help() {
    echo -e "The setup-pib.sh script has two execution modes:"
    echo -e "(normal mode and development mode)""$NEW_LINE"
    echo -e "$INFO""Normal mode (don't add any arguments or options)""$RESET_TEXT_COLOR"
    echo -e "$INFO""If you are do not know what the flags for development mode do, use the normal mode""$RESET_TEXT_COLOR"
    echo -e "Example: ./setup-pib""$NEW_LINE"
    echo -e "$INFO""Development mode (specify the branches you want to install)""$RESET_TEXT_COLOR"

    echo -e "You can either use the short or verbose command versions:"
    echo -e "-f=YourBranchName or --frontend-branch=YourBranchName"
    echo -e "-b=YourBranchName or --backend-branch=YourBranchName"
    echo -e "-l or --local for a local installation of the software over using a containerized setup using Docker"
    echo -e "-r or --reset to reset all installation status and start fresh"

    echo -e "$NEW_LINE""Examples:"
    echo -e "    ./setup-pib -b=main -f=PR-566"
    echo -e "    ./setup-pib --backend-branch=main --frontend-branch=PR-566"
    echo -e "    ./setup-pib --reset"

    exit
}

# ---------- SETUP STARTS FROM HERE -----------

# Reduplicate output to an extra log file as well
LOG_FILE="$HOME/setup-pib.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Hello $USER! We start the setup by allowing you permanently to run commands with admin-privileges. This change is reverted at the end of the setup."
if [[ "$(id)" == *"(sudo)"* ]]; then
    echo "For this change please enter your password..."
    sudo bash -c "echo '$USER ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/$USER"
else
    echo "For this change please enter the root-password. It is most likely just your normal one..."
    su root bash -c "usermod -aG sudo $USER ; echo '$USER ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/$USER"
fi

DISTRIBUTION=$(get_distribution) # e.g., 'ubuntu'
export DISTRIBUTION
DIST_VERSION=$(get_dist_version "$DISTRIBUTION")  # e.g., 'jammy'
export DIST_VERSION
print INFO "$DISTRIBUTION $DIST_VERSION"

# VALIDATE CLI ARGUMENTS
BRANCH_BACKEND="main"
BRANCH_FRONTEND="main"
INSTALL_METHOD="docker"
RESET_INSTALL=false

# Check if branch was specified
while [ $# -gt 0 ]; do
    case "$1" in
        -f=* | --frontend-branch=*)
            BRANCH_FRONTEND="${1#*=}"
            ;;
        -b=* | --backend-branch=*)
            BRANCH_BACKEND="${1#*=}"
            ;;
        -l | --legacy)
            INSTALL_METHOD="legacy"
            ;;
        -r | --reset)
            RESET_INSTALL=true
            ;;
        -h | --help)
            show_help
            ;;
        *)
            print ERROR "invalid input options"
            show_help
            ;;
    esac
    shift
done

# Clear previous installation status if requested
if [ "$RESET_INSTALL" = true ]; then
    if [ -d "$STATUS_DIR" ]; then
        print INFO "Resetting installation status"
        rm -rf "$STATUS_DIR"
    fi
fi

# Create status directory if it doesn't exist
mkdir -p "$STATUS_DIR"

# Run installation steps
remove_apps || print WARN "Failed to remove default software"
install_system_packages || { print ERROR "Failed to install system packages"; exit 1; }
disable_power_notification || print WARN "Failed to disable power notifications"
clone_repositories || { print ERROR "Failed to clone repositories"; exit 1; }
move_setup_files || print WARN "Failed to move setup files"
install_DBbrowser || print WARN "Failed to install DB browser"
install_BrickV || print WARN "Failed to install Brick viewer"
set_system_settings || print INFO "Skipped setting system settings"

print INFO "Installation method: ${INSTALL_METHOD}"
if [ "$INSTALL_METHOD" = "legacy" ]; then
    print INFO "Going to install Cerebra locally (LEGACY MODE NOT WORKING ON RASPBERRY PI 5)"
    install_local || print ERROR "Failed to install Cerebra locally"
else
    print INFO "Going to install Cerebra via Docker"
    install_docker || print ERROR "Failed to install Cerebra via Docker"
fi

cleanup

# Check if all required steps completed successfully
REQUIRED_STEPS=("install_system_packages" "clone_repositories")
ALL_REQUIRED_COMPLETED=true

for step in "${REQUIRED_STEPS[@]}"; do
    if ! is_step_completed "$step"; then
        print ERROR "Required step '$step' was not completed successfully"
        ALL_REQUIRED_COMPLETED=false
    fi
done

if [ "$ALL_REQUIRED_COMPLETED" = true ]; then
    print SUCCESS "Finished installation, for more information on how to use pib and Cerebra, visit https://pib-rocks.atlassian.net/wiki/spaces/kb/overview?homepageId=65077450"
    print SUCCESS "Reboot pib to apply all changes"
else
    print ERROR "Installation did not complete successfully. Please run the script again with --reset option to retry all steps"
    exit 1
fi
