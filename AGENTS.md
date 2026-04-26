always activate the virtual environment in ./env before trying to run any python, never create a new virtualenv

before every code change, run this:
`git add .; git commit -m"before AI agent {short_update_name} update. agent: {your name}"; git push`
let short update name not be longer than 3 words
don't worry if push fails, it's the commit that's important

after every edit turn do `git add .` and make a long commit exhaustively explain every change you made in a detailed easy to understand way. then push; now this push is important, make sure it works

never add `.onnx` files to `.gitignore`.

be minimalist