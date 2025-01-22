git filter-branch --env-filter '
OLD_EMAIL="serhii.hamotskyi@hs-anhalt.de"
OLD_NAME="Serhii Hamotskyi"
NEW_EMAIL="serhii@serhii.net"

if [ "$GIT_COMMITTER_EMAIL" = "$CORRECT_EMAIL" ] && [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$OLD_NAME"
    export GIT_COMMITTER_EMAIL="$OLD_EMAIL"
    export GIT_AUTHOR_NAME="$OLD_NAME"
    export GIT_AUTHOR_EMAIL="$OLD_EMAIL"    
fi'
