-- Layer to create quests and act as middle-man between Evennia and Agent
require 'utils'
local _ = require "underscore"

local DEBUG = true

local DEFAULT_REWARD = -0.01
local JUNK_CMD_REWARD = -0.1
local STEP_COUNT = 0 -- count the number of steps in current episode

--Simple quests
quests = {'you are hungry','you are sleepy', 'you are bored', 'you are getting fat'}
quests_mislead = {
    'You are not hungry','You are not sleepy','You are not bored', 'You are not getting fat'}

quest_actions = {'eat', 'sleep', 'watch' ,'exercise'} -- aligned to quests above

current_quest = 0
current_mislead = 0

rooms = {'Living', 'Garden', 'Kitchen','Bedroom'}

actions = {"eat", "sleep", "watch", "exercise", "go"}
objects = {'north','south','east','west'}
-- read the rest of the obejcts from build file
-- order in build file: tv, bike, apple, bed

extra_vocab = {'not','but', 'now'} -- words that are necessary for initial vocab building but not in other text
symbols = {}
symbol_mapping = {}

local descriptions = {}

local NUM_ROOMS = 4
local state_dim = 0

local current_room_description = ""

function random_teleport()
    local room_index = torch.random(1, NUM_ROOMS)
    data_out('@tel tut#0'..room_index)
    sleep(0.1)
    data_in()
    data_out('l')
    if DEBUG then
        print('Start Room : ' .. room_index ..' ' .. rooms[room_index])
    end
end

function get_quest_text(quest_num, mislead_num)
    quest_num = quest_num or current_quest
    mislead_num = mislead_num or current_mislead
    return quests_mislead[mislead_num] .. ' now but ' .. quests[quest_num] ..' now.'
end

function getAllStates()
    -- ``descriptions`` needs to have been populated
    local states = {}

    for q=1,#quests do
        for m=1,#quests_mislead do
            for d=1,#descriptions do
                local state = descriptions[d]
                state = state .. " " .. get_quest_text(q, m)
                table.insert(states, state)
            end
        end
    end

    return states
end

function random_quest()
    -- Pick a random subset of goals to make a quest.
    -- QUEST_LEVELS controls the number of goals in the quest.
    indices = torch.randperm(#quests)
    current_quest = indices[1]
    current_mislead = indices[#quests]

    if DEBUG then
        print(
            "Start quest",
            get_quest_text(),
            quest_actions[current_quest])
    end
end

function login(user, password)
    local num_rooms = 4
    local pre_login_text = data_in()
    print(pre_login_text)
    sleep(1)
    data_out('connect ' .. user .. ' ' .. password)
end

--Function to parse the output of the game (to extract rewards, etc. )
function parse_game_output(text)
    -- extract REWARD if it exists
    -- text is a list of sentences
    local reward = nil
    local text_to_agent = {current_room_description, get_quest_text()}
    for i=1, #text do
        if i < #text  and string.match(text[i], '<EOM>') then
            text_to_agent = {current_room_description, get_quest_text()}
        elseif string.match(text[i], "REWARD") then
            if string.match(text[i], quest_actions[current_quest]) then
                reward = tonumber(string.match(text[i], "%d+"))
            end
        elseif string.match(text[i], 'not available') or string.match(text[i], 'not find') then
                reward = JUNK_CMD_REWARD
        end
    end

    if not reward then
        reward = DEFAULT_REWARD
    end

    return text_to_agent, reward
end


--take a step in the game
function step_game(action_index, object_index, gameLogger)
    local command = build_command(actions[action_index], objects[object_index], gameLogger)
    data_out(command)
    if DEBUG then
        print(actions[action_index] .. ' ' .. objects[object_index])
    end
    STEP_COUNT = STEP_COUNT + 1
    return getState(gameLogger)
end

-- starts a new game
function newGame(gameLogger)

    STEP_COUNT = 0
    random_teleport()
    random_quest()

    if gameLogger then
    end

    return getState(gameLogger)
end

-- build game command to send to the game
function build_command(action, object, logger)
    if logger then
        logger:write(">>" .. action .. ' '.. object..'\n')
    end

    return action .. ' ' ..object
end


-- Modifies ``symbols`` and ``symbol_mapping``.
-- Add the words in ``list_words`` to ``symbols`` and ``symbol_mapping``.
-- IMP: make sure we're using simple english - ignores punctuation, etc.
function parseLine(list_words, start_index)
    local sindx
    start_index = start_index or 1
    for i=start_index,#list_words do
        word = split(list_words[i], "%a+")[1]
        word = word:lower()
        if symbol_mapping[word] == nil then
            sindx = #symbols + 1
            symbols[sindx] = word
            symbol_mapping[word] = sindx
        end
    end
end

-- Modifies ``symbols`` and ``symbol_mapping``.
-- Here we're adding words to the from the quests
-- that we've predefined.
function addQuestWordsToVocab()
    for i, quest in pairs(quests) do
        parseLine(split(quest, "%a+"))
    end
end

-- Modifies ``symbols`` and ``symbol_mapping``.
-- Here we're adding words the ``extra_vocab`` table
function addExtraWordsToVocab()
    for i, word in pairs(extra_vocab) do
        word = word:lower()
        if symbol_mapping[word] == nil then
            sindx = #symbols + 1
            symbols[sindx] = word
            symbol_mapping[word] = sindx
        end
    end
end

function addDescription(words, start_index)
    local sindx
    start_index = start_index or 1

    desc = {}

    for i=start_index,#list_words do
        table.insert(desc, words[i])
    end

    table.insert(descriptions, table.concat(desc, " "))
end

-- read in text data from ``filename``- nicely tokenized
-- read all lines that start with @detail or @desc
-- also extracts actionable objects from @create/drop lines
-- see evennia/contrib/text_sims/build.ev for an example
-- called from the main agent script
function makeSymbolMapping(filename)
    local file = io.open(filename, "r");
    local data = {}
    local parts
    print (filename)
    for line in file:lines() do
        list_words = split(line, "%S+")
        if list_words[1] == '@detail' or list_words[1] == '@desc' then
            parseLine(list_words, 4)
            addDescription(list_words, 4)
        elseif list_words[1] == '@create/drop' then
            -- add to actionable objects
            table.insert(objects, split(list_words[2], "%a+")[1])
        end
    end

    addQuestWordsToVocab()
    addExtraWordsToVocab()

    all_states = getAllStates()
    state_dim = torch.max(torch.Tensor(_.map(all_states, function (s) return #s end)))
end

local function getStateDim()
    return state_dim
end

function getState(logger, print_on, as_string)
    local terminal = (STEP_COUNT >= MAX_STEPS)
    local inData = data_in()
    while #inData == 0 or not string.match(inData[#inData],'<EOM>') do
        TableConcat(inData, data_in())
    end

    data_out('look')
    local inData2 = data_in()
    while #inData2 == 0 or not string.match(inData2[#inData2],'<EOM>') do
        TableConcat(inData2, data_in())
    end
    current_room_description = inData2[1]

    print("Raw game text: ")
    print(inData)
    print("Room desc: " .. current_room_description)

    local text, reward = parse_game_output(inData)
    text = table.concat(text, ' ')
    if DEBUG or print_on then
        print("Text for agent: ")
        print(text)
        print("Reward received: ")
        print(reward)

        sleep(0.1)
        if reward > 0 then
            print(text, reward)
            sleep(2)
        end
    end

    if reward >= 1 then
        --quest has been succesfully finished
        current_quest = 0
        current_mislead = 0
        terminal = true
    end

    if logger then
        logger:write(text, '\n')
        logger:write('Reward: ' .. reward, '\n')
        if terminal then
            logger:write('****************************\n\n')
        end
    end

    if not as_string then
        text = #text < state_dim and text or text:sub(1, state_dim)
        text = text .. string.rep(' ', state_dim - #text)
        text = torch.ByteTensor(torch.ByteStorage():string(text))
    end

    return text, reward, terminal
end

function getActions()
    return actions
end

function getObjects()
    return objects
end


return {
    makeSymbolMapping = makeSymbolMapping,
    getStateDim = getStateDim,
    getAllStrings = getAllStrings,
    getActions = getActions,
    getObjects = getObjects,
    getState = getState,
    step = step_game,
    newGame = newGame,
    nextRandomGame = nextRandomGame,
    vector_function = vector_function
}