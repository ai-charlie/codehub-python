#!/bin/sh
dir_root="$HOME/workspace"
echo $dir_root
cd $dir_root
dir=`ls -d */*/*/ && ls -d */*/ && ls -d */`
out=$dir_root'/git_remote_url.md'
touch $out
echo "# GIT REMOTE URL" > $out
function get_git_remote_url {
for table in $dir
do
if [ -d "$table/.git" ]; then
cd $table
# pwd
git remote -v | grep -m 1 -o  "$1.*\.git" >> $out
cd $dir_root
fi
done
}
get_git_remote_url "git@github"
get_git_remote_url "https://github.com"
get_git_remote_url "http://github.com"
get_git_remote_url "git@[0-9]"
get_git_remote_url "https://[0-9]"
get_git_remote_url "http://[0-9]"
sed -i s#"https://github.com/"#"git@github.com:"#g $out
sed -i s#"http://github.com/"#"git@github.com:"#g $out

