#!/bin/bash

. ./deploy.sh --source-only

TAG=$1

if [[ -z $TAG ]]
then
    echo "TAG not provided"
    exit 0
fi

# Get tagged instances
instance_ids=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
        "Name=tag:$TAG_KEY,Values=$TAG" \
    --query "Reservations[].Instances[].InstanceId")

create_response_array "$instance_ids"

for i in "${response_array[@]}"
do
  echo "InctanceID: $i"
	echo "Getting DNS..."
	dns_name=$(aws ec2 describe-instances --instance-ids "$i" \
		  --query "Reservations[].Instances[].PublicDnsName" --output text)
  echo "$dns_name"
	
	echo "Collecting output..."
	dir_out="$OUTPUT_LOCATION/output_$TAG-$i"
	mkdir $dir_out
	scp -i $CERT_FILE "ec2-user@$dns_name:$REMOTE_LOCATION/core/output/output.csv" \
		"$dir_out/output_$TAG-$i.csv"

	# collect logs from instance
	echo "Collecting logs..."
	dir_log="$LOGS_LOCATION/log_$TAG-$i"
	mkdir $dir_log
	scp -i $CERT_FILE "ec2-user@$dns_name:$REMOTE_LOCATION/core/logs/log.txt" \
		"$dir_log/log_$TAG-$i.txt"
done

echo "Done!"
