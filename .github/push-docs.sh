#!/bin/bash
set -eu

[ "$GH_PASSWORD" ] || exit 12

sitemap() {
    WEBSITE='https://nansencenter.github.io/DAPPER'
    find -name '*.html' |
        sed "s,^\.,$WEBSITE," |
        sed 's/index.html$//' |
        grep -v '/google.*\.html$' |
        sort -u  > 'sitemap.txt'
    echo "Sitemap: $WEBSITE/sitemap.txt" > 'robots.txt'
}

# The docs were generated from master, but here we need the gh-pages branch
git clone -b gh-pages "https://patricknraanes:$GH_PASSWORD@github.com/$GITHUB_REPOSITORY.git" gh-pages
cp -R docs/* gh-pages/docs/
cd gh-pages
# sitemap
git add *
if git diff --staged --quiet; then
  echo "$0: No changes to commit."
  exit 0
fi

if ! git config user.name; then
    git config user.name 'github-actions'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
fi

head=$(git rev-parse --short HEAD)
git commit -a -m "CI: Update docs for $head"
git push
