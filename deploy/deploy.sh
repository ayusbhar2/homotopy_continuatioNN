#!/bin/bash

# Add the following entries to your ~/.bash_profile

# export CERT_FILE="/full/path/to/cert_file.cer"
# export IMAGE_ID="<amazon-image-id>"
# export KEY_NAME="<ec2 key-name>"
# export LOGS_LOCATION="<full-path-to-directory>/homotopy_continuation/deploy/cloud_logs"
# export OUTPUT_LOCATION="<full-path-to-directory>/homotopy_continuation/deploy/cloud_output"
# export REMOTE_LOCATION="~"
# export REPO_LOCATION="<full-path-to-directory>/homotopy_continuation"
# export REPO_NAME="core"
# export SEC_GRP="<IAM-security-group-name>"
# export TAG_KEY="<whatever-you-want>"
# export TAR_FILE_NAME="core.tar.gz"



# Function to generate an array from aws response string
# NOTE: this func is prbly redundant with the appropriate aws cli options
create_response_array () {
	str=$(awk '{ sub(/\[\s*/, ""); sub(/\s*\]/, ""); print }' <<< $1) # remove brackets
	element_names=(`echo $str | tr ',' ' '`) # split into separate element names
	arr=""
	for element in "${element_names[@]}"
	do
		e=$(awk '{sub(/\"/, ""); sub(/\"/, ""); print }' <<< $element) # remove quotes
		arr=$(echo "$arr,$e")
	done
	arr=${arr:1} # remove the leading comma
	response_array=(`echo $arr | tr ',' ' '`) # recreate the array without quotes
}

echo "REMINDER: Ensure that the setup file has the right parameters.."

if [[ "$1" != "--source-only" ]]
then
	TYPE=$1 # c5.9xlarge (36 thr), c5.4xlarge (16 thr)
	COUNT=$2
	TAG=$3 # di2H1m1dx2dy5_noconv

	if [ -z "$1" ]
	then
		echo "TYPE not provided..."
		exit 0
	fi

	if [ -z "$2" ]
	then
		echo "COUNT not provided..."
		exit 0
	fi

	if [ -z "$3" ]
	then
		echo "TAG not provided..."
		exit 0
	fi

	
    # Create instances
	echo "Creating $COUNT instances of type $TYPE..."
	aws ec2 run-instances --image-id $IMAGE_ID \
		--count $COUNT --instance-type $TYPE --key-name $KEY_NAME \
		--security-groups $SEC_GRP \
		--tag-specifications "ResourceType=instance,Tags=[{Key=$TAG_KEY,Value=$TAG}]"

	
	# Check if all instances are up and running
	instance_ids=$(aws ec2 describe-instances \
		--filters "Name=instance-state-name,Values=running" \
			"Name=tag:$TAG_KEY,Values=$TAG" \
		--query "Reservations[].Instances[].InstanceId")

	create_response_array "$instance_ids"

	l=${#response_array[@]}

	if [[ $l -eq $COUNT ]]
	then
		echo "$l instances up and running!"
	else
		while [[ $l -lt $COUNT ]]
		do
			sleep 10
			echo "Checking if all instances are up and running..."
		   	instance_ids=$(aws ec2 describe-instances \
				--filters "Name=instance-state-name,Values=running" \
					"Name=tag:$TAG_KEY,Values=$TAG" \
				--query "Reservations[].Instances[].InstanceId")

			create_response_array "$instance_ids"
			l=${#response_array[@]}
		done
		echo "$l instances up and running!"
	fi


	# Check if all instances are reachable
	echo "Checking if all instances are reachable..."

	pass_count=0
	while [[ $pass_count -lt  $COUNT ]]
	do
		for instance_id in "${response_array[@]}"
		do
			echo "checking $instance_id..."
			status=$(aws ec2 describe-instance-status --instance-id  $instance_id\
				     --query "InstanceStatuses[].InstanceStatus[].Status" --output="text")
			if [[ $status == "ok" ]]
			then
				echo "$instance_id is reachable"
				((pass_count=pass_count+1))
			fi
		done
		sleep 10
	done


	# Archive code to deploy
	echo "Archiving code for deployment..."
	tar czf $TAR_FILE_NAME -C $REPO_LOCATION  $REPO_NAME


	# Deploy code to each instance
	for instance_id in "${response_array[@]}"
	do
		# Get DNS names
		echo "Getting DNS name for $instance_id..."
		dns_name=$(aws ec2 describe-instances \
			--instance-ids "$instance_id"\
			--query "Reservations[].Instances[].PublicDnsName"\
			--output="text")
		echo "DNS name: $dns_name"

		# Transfer files to instance
		echo "Transferring files..."
		cat $TAR_FILE_NAME | ssh -i $CERT_FILE -o StrictHostKeyChecking=no "ec2-user@$dns_name" "tar xzf - -C $REMOTE_LOCATION/"

		# Launch setup script on instance
		# TODO: set TCPKeepAlive in /etc/ssh/sshd_config file
		echo "Launching setup..."
		ssh -i $CERT_FILE "ec2-user@$dns_name" 'bash -s' < setup.sh &
		echo "Launch Done! To follow the progress, ssh into the innstance(s)."
	done


fi
