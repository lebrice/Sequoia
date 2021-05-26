all_ids=$(eai job ls --state queuing --fields id --no-header)
for id in $all_ids
do
  eai job kill $id
done