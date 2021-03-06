require 'torch'
require 'nn'
require '../utils'

do
    local Text2Vector, _ = torch.class('Text2Vector', 'nn.Module')

    function Text2Vector:__init(dimension, use_cache)
        self.cache = use_cache and {}
        self.output = torch.Tensor()
        self.dimension = dimension
    end

    function Text2Vector:fillVector(text, vector)
        error("``fillVector`` is an abstract method that needs to " ..
              "be overridden by sub-classes.")
    end

    function Text2Vector:preprocessText(text)
        if torch.typename(text) == "torch.ByteTensor" then
            text = text:storage():string()
        elseif torch.typename(text) == "torch.FloatTensor" then
            text = text:byte():storage():string()
        end

        assert(type(text) == 'string',
               'Input must be string or a ByteTensor encoding a string.')
        return text
    end

    function Text2Vector:handleInstance(row, loc, text)
        local cached

        text = self:preprocessText(text)
        cached = self.cache and self.cache[text]
        if cached then
            loc:indexCopy(1, torch.LongTensor{row}, cached)
        else
            self:fillVector(text, loc[row])
            if self.cache then
                self.cache[text] = loc[row]:view(1, self.dimension):clone()
            end
        end
    end

    function Text2Vector:updateOutput(text)
        local typename, size

        typename = torch.typename(text)
        if type(text) == 'table' or typename and text:dim() > 1 then
            -- Batch mode
            size = type(text) == 'table' and #text or text:size(1)
            self.output:resize(size, self.dimension):zero()
            for i=1,size do
                self:handleInstance(i, self.output, text[i])
            end
        elseif typename and text:dim() == 1 or type(text) == 'string' then
            self.output:resize(1, self.dimension):zero()
            self:handleInstance(1, self.output, text)
            self.output = self.output:reshape(self.dimension)
        else
            error("Invalid input type. Expected vector, matrix, string or array of strings.")
        end

        return self.output
    end
end

do
    local Lookup, _ = torch.class('Lookup', 'Text2Vector')

    function Lookup:__init(all_states)
        local use_cache = false
        Text2Vector.__init(self, #all_states, use_cache)

        self.max_state = 0
        self.all_states = {}
    end

    function Lookup:fillVector(text, vector)
        text = string.gsub(text, "{.","")

        if not self.all_states[text] then 
            self.max_state = self.max_state + 1
            self.all_states[text] = self.max_state
        end

        vector[self.all_states[text]] = 1.0
    end
end

local function make_lookup(all_states)
    return Lookup(all_states), #all_states
end

do
    local BagOfWords, _ = torch.class('BagOfWords', 'Text2Vector')

    function BagOfWords:__init(symbols, symbol_mapping, use_cache)
        Text2Vector.__init(self, #symbols, use_cache)

        self.symbols = symbols
        self.symbol_mapping = symbol_mapping
    end

    function BagOfWords:fillVector(text, vector)
        local idx

        text = string.gsub(text, "{.","")
        local words = split(text, "%a+")

        for i, word in ipairs(words) do
            word = word:lower()

            if symbol_mapping[word] then
                idx = symbol_mapping[word]
                vector[idx] = vector[idx] + 1
            end
        end
    end
end

local function make_bow(symbols, symbol_mapping)
    return BagOfWords(symbols, symbol_mapping), #symbols
end


do
    local BagOfBigrams, _ = torch.class('BagOfBigrams', 'Text2Vector')

    function BagOfBigrams:__init(symbols, symbol_mapping, size, use_cache)
        local dimension = size and #symbols * size or #symbols * #symbols
        Text2Vector.__init(self, dimension, use_cache)

        self.symbols = symbols
        self.symbol_mapping = symbol_mapping

        self.bigram_mapping = {}
        self.bigrams = {}

        self.n_bigrams = 0
    end

    function BagOfBigrams:fillVector(text, vector)
        local words, idx, bigram, next_word

        text = string.gsub(text, "{.","")
        words = split(text, "%a+")

        for i, word in ipairs(words) do
            if i == #words then
                break
            end

            word = word:lower()
            next_word = words[i+1]:lower()
            bigram = word .. ' ' .. next_word

            idx = self.bigram_mapping[bigram]

            if not idx then
                if self.n_bigrams < self.dimension then
                    if self.symbol_mapping[word] and self.symbol_mapping[next_word] then
                        table.insert(self.bigrams, bigram)
                        self.n_bigrams = #self.bigrams
                        idx = self.n_bigrams

                        self.bigram_mapping[bigram] = idx
                        vector[idx] = vector[idx] + 1
                    end
                end
            else
                vector[idx] = vector[idx]  + 1
            end
        end
    end
end

local function make_bob(symbols, symbol_mapping, size)
    local bob = BagOfBigrams(symbols, symbol_mapping, size)
    return bob, bob.dimension
end


do
    local OrderedList, _ = torch.class('OrderedList', 'Text2Vector')

    function OrderedList:__init(symbols, symbol_mapping, dimension, use_cache)
        Text2Vector.__init(self, dimension, use_cache)

        self.symbols = symbols
        self.symbol_mapping = symbol_mapping

        self.reverse = true
        self.null_index = #self.symbols + 1
    end

    function OrderedList:fillVector(text, vector)
        local idx

        vector:fill(self.null_index)

        text = string.gsub(text, "{.","")
        local words = split(text, "%a+")

        for i, word in ipairs(words) do
            word = word:lower()
            if self.reverse then
                idx = self.dimension + 1 - i
            else
                idx = i
            end

            if self.symbol_mapping[word] then
                vector[idx] = self.symbol_mapping[word]
            end
        end
    end
end

local function make_ordered_list(symbols, symbol_mapping, dimension)
    local ol = OrderedList(symbols, symbol_mapping, dimension)
    return ol, ol.dimension
end


do
    local RandomVector, _ = torch.class('RandomVector', 'Text2Vector')

    function RandomVector:__init(rv_func, dimension)
        local use_cache = true
        Text2Vector.__init(self, dimension, use_cache)

        self.rv_func = rv_func
    end

    function RandomVector:fillVector(text, vector)
        vector:copy(self.rv_func(self.dimension))
    end
end

local function make_random(rv_func, dimension)
    local rv = RandomVector(rv_func, dimension)
    return rv, rv.dimension
end


local function test_bow()
    print("Testing BoW...")

    symbols = {'a', 'b', 'c'}
    symbol_mapping = {a=1, b=2, c=3}
    b, s = make_bow(symbols, symbol_mapping)
    assert(b:forward("a b"):eq(torch.Tensor{1, 1, 0}):all())

    -- Batch mode
    result = torch.Tensor{{1, 1, 1}, {1, 2, 1}}
    assert(b:forward{"c b a", "a, b, b, c"}:eq(result):all())

    -- Part of a network
    seq = (
        nn.Sequential()
        :add(b)
        :add(nn.Linear(s, 1)))

    seq:get(2).weight = torch.Tensor{{1, 1, 1}}
    seq:get(2).bias = torch.Tensor{3}

    assert(seq:forward("a b"):eq(torch.Tensor{5}):all())
    seq:backward("a b", torch.Tensor{1})

    assert(seq:forward{"a b", "b c"}:eq(torch.Tensor{5, 5}):all())
    seq:backward({"a b", "b c"}, torch.Tensor{{1}, {1}})
    print("Passed :-).")
end

local function test_bob()
    print("Testing BoB...")

    symbols = {'a', 'b', 'c'}
    symbol_mapping = {a=1, b=2, c=3}
    b, s = make_bob(symbols, symbol_mapping)
    assert(b:forward("a b"):eq(torch.Tensor{1, 0, 0, 0, 0, 0, 0, 0, 0}):all())

    -- Batch mode
    result = torch.Tensor{{0, 1, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 0, 0}}
    assert(b:forward{"c b a", "a, b, b, c"}:eq(result):all())

    -- Part of a network
    seq = (
        nn.Sequential()
        :add(b)
        :add(nn.Linear(s, 1)))

    seq:get(2).weight = torch.Tensor():resize(1, 9):fill(1)
    seq:get(2).bias = torch.Tensor{3}

    assert(seq:forward("a b"):eq(torch.Tensor{4}):all())
    seq:backward("a b", torch.Tensor{1})

    assert(seq:forward{"a b", "b c"}:eq(torch.Tensor{4, 4}):all())
    seq:backward({"a b", "b c"}, torch.Tensor{{1}, {1}})
    print("Passed :-).")
end

local function test_ol()
    print("Testing OrderedList...")

    symbols = {'a', 'b', 'c'}
    symbol_mapping = {a=1, b=2, c=3}
    b, s = make_ordered_list(symbols, symbol_mapping, 4)
    assert(b:forward("b"):eq(torch.Tensor{4, 4, 4, 2}):all())
    assert(b:forward("a b"):eq(torch.Tensor{4, 4, 2, 1}):all())
    assert(b:forward("c b c"):eq(torch.Tensor{4, 3, 2, 3}):all())
    assert(b:forward("c b a"):eq(torch.Tensor{4, 1, 2, 3}):all())

    -- Batch mode
    result = torch.Tensor{{4, 1, 2, 3}, {3, 2, 2, 1}}
    assert(b:forward{"c b a", "a, b, b, c"}:eq(result):all())

    -- Part of a network
    seq = (
        nn.Sequential()
        :add(b)
        :add(nn.Linear(s, 1)))

    seq:get(2).weight = torch.Tensor{{1, 1, 1, 1}}
    seq:get(2).bias = torch.Tensor{3}

    assert(seq:forward("a b"):eq(torch.Tensor{14}):all())
    seq:backward("a b", torch.Tensor{1})

    assert(seq:forward{"a b", "b c"}:eq(torch.Tensor{14, 16}):all())
    seq:backward({"a b", "b c"}, torch.Tensor{{1}, {1}})
    print("Passed :-).")
end

local function test_random_vector()
    print("Testing RandomVector...")

    local random_vec_func = (
        function (size)
            return torch.rand(size) - 0.5
        end
    )

    dimension = 10
    b, s = make_random(random_vec_func, dimension)
    assert(b:forward("b"):eq(b:forward("b"):clone()):all())
    assert(not b:forward("b"):clone():eq(b:forward("b c"):clone()):all())

    -- Batch mode
    b:forward{"c b a", "a, b, b, c"}

    -- Part of a network
    seq = (
        nn.Sequential()
        :add(b)
        :add(nn.Linear(s, 1)))

    seq:forward("a b")
    seq:backward("a b", torch.Tensor{1})

    seq:forward{"a b", "b c"}
    seq:backward({"a b", "b c"}, torch.Tensor{{1}, {1}})
    print("Passed :-).")
end

if arg[1] == 'test' then
    test_bow()
    test_bob()
    test_ol()
    test_random_vector()
end

text_to_vector = {
    Text2Vector=Text2Vector,

    Lookup=Lookup,
    make_lookup=make_lookup,

    BagOfWords=BagOfWords,
    make_bow=make_bow,

    BagOfBigrams=BagOfBigrams,
    make_bob=make_bob,

    OrderedList=OrderedList,
    make_ordered_list=make_ordered_list,

    RandomVector=RandomVector,
    make_random=make_random
}

return text_to_vector
