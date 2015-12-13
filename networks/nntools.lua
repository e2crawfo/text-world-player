local function filter(t, f)
    local out = {}

    for k, v in pairs(t) do
        if f(k, v) then out[k] = v end
    end

    return out
end


local function update_table(base, new, overwrite)
    overwrite = overwrite or true
    for k, v in pairs(new) do
        if not base[k] or overwrite then
            base[k] = v
        end
    end
end

function table_length(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

local function clone_net(net)
    return net:clone('weight', 'bias', 'gradWeight', 'gradBias')
end

local function get_all_words(text)
    local parse_trees, pts, all_words
    parse_trees = parse.parse_sentences({text})

    pts = {}
    for s, trees in pairs(parse_trees) do
        pts[s] = trees[1]
    end

    all_words = {}
    for s, tree in pairs(pts) do
        for w, v in pairs(tree:get_words()) do
            all_words[w] = v
        end
    end

    return all_words
end

nntools = {
    filter=filter,
    update_table=update_table,
    table_length=table_length,
    clone_net=clone_net,
    get_all_words=get_all_words,
}

return nntools
