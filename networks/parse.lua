local BinaryParseTree = {
    word = nil,
    left = nil,
    right = nil,
    label = nil,
    terminal = true
}

function BinaryParseTree:new (o)
    o = o or {}   -- create object if user does not provide one
    setmetatable(o, self)
    self.__index = self
    return o
end


function BinaryParseTree:get_subtrees ()
    -- Returns a table mapping from subtrees to 1 - essentially a set.
    local subtrees = {}

    local function f (tree)
        subtrees[tree] = 1
        if not tree.terminal then
            f(tree.left)
            f(tree.right)
        end
    end

    f(self)
    return subtrees
end


function BinaryParseTree:get_words ()
    -- Returns a table mapping from words to 1 - essentially a set.
    local words = {}

    local function f (tree)
        if tree.terminal then
            words[tree.word] = 1
        else
            f(tree.left)
            f(tree.right)
        end
    end

    f(self)
    return words
end


function BinaryParseTree:get_text ()
    local words = {}

    local function f (tree)
        if tree.terminal then
            words[#words + 1] = tree.word
        else
            f(tree.left)
            f(tree.right)
        end
    end

    f(self)
    return table.concat(words, ' ')
end


function BinaryParseTree:__tostring ()
    local function f (tree)
        if tree.terminal then
            return string.format("(%s %s)", tree.label, tree.word)
        else
            return string.format("(%s %s %s)", tree.label, f(tree.left), f(tree.right))
        end
    end

    return f(self)
end


local function string_to_parse_tree (s)
    -- Assumes parentheses are balanced.
    -- Assumes parse tree is binary.

    local tree, left_string, right_string, children

    if s:sub(1, 1) == "(" then
        s = s:sub(2, -2)
    end

    tree = BinaryParseTree:new{}

    local split_space = s:gmatch("%S+")
    tree.label = split_space()

    children = s:gmatch("%b()")
    left_string = children()
    right_string = children()

    if right_string ~= nil then
        tree.left = string_to_parse_tree(left_string)
        tree.right = string_to_parse_tree(right_string)
        tree.terminal = false
        tree.word = nil
    else
        tree.left = nil
        tree.right = nil
        tree.terminal = true
        tree.word = split_space()
    end

    return tree
end


local function map(tbl, f)
    local t = {}
    for k,v in pairs(tbl) do
        t[k] = f(v)
    end
    return t
end


local surround = function (s) return "'" .. s .. "'" end


local function parse_sentences (sentences)
    -- sentences: array of strings
    -- Each string could contain multiple sentences. We
    -- leave it up to parse.py to segment the sentences for us,
    -- and return separate parse trees for each sentence found.

    sentences = map(sentences, surround)
    local args = table.concat(sentences, " ")
    local parse_script = "/home/eric/Dropbox/classes/comp599/project/text-world-player/networks/parse.py"

    -- Parse sentences by calling out to parse.py
    local handle = io.popen("python " .. parse_script .. " " .. args)

    local new_sentences = {}
    local parses = {}
    local i = 0
    local line = handle:read()
    while line do
        if tonumber(line) ~= nil then
            i = i + 1
            parses[i] = {}

            table.insert(new_sentences, handle:read())
        else
            table.insert(parses[i], line)
        end

        line = handle:read()
    end
    handle:close()

    -- Turn strings returned from parse.py into tree-like tables
    local trees = {}

    for i, ps in ipairs(parses) do
        trees[new_sentences[i]] = {}

        for j, p in ipairs(ps) do
            tree = string_to_parse_tree(p)
            table.insert(trees[new_sentences[i]], tree)
        end
    end

    return trees
end


if false then
    sentences = {"I like ham", "The dog won the tournament"}
    all_trees = parse_sentences(sentences)

    for s, trees in pairs(all_trees) do
        print("Parses for sentence: ", s)
        for j, t in ipairs(trees) do
            print(t)
        end
    end
end

parse = {
    BinaryParseTree=BinaryParseTree,
    parse_sentences=parse_sentences
}