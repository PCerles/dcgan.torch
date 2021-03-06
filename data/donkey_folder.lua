--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'io'
require 'paths'
paths.dofile('dataset.lua')
vigo = require 'csvigo'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '/data/local/imagenet-fetch/256'
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

   --load captionI
   local caption_path = path:gsub('dataSmall2/images', 'captionsSmallVec2')
   caption_path = caption_path:gsub('jpg','csv')
   local caption = 'NONE'
   if paths.filep(caption_path) then
	caption = torch.load(caption_path)
   end
   -- load caption_rep
   --local caption_rep_path = path:gsub('data/images', 'caption_vecs')
   --caption_rep_path = caption_rep_path:gsub('jpg','csv')
   --local caption_rep = 'NONE'

  -- if paths.filep(caption_path) then
    -- local caption_file = io.open(caption_path)
     --caption = caption_file:read('*a')
     --caption_file:close()
  -- end 

   --if paths.filep(caption_rep_path) then
   --  local temp_vec = vigo.load(caption_rep_path)
   --  for k,v in pairs(temp_vec) do temp_key = k end
   --  local temp_table = {}
   --  table.insert(temp_table,temp_key)
   --  for k,v in pairs(temp_vec[temp_key]) do table.insert(temp_table,v) end
   --  caption_rep = torch.Tensor(temp_table) 
   -- end

   -- load caption vocab rep
   local vocab_rep_path = path:gsub('dataSmall2/images', 'vocab_vecs2')
   vocab_rep_path = vocab_rep_path:gsub('jpg','csv')
   local vocab_rep = 'NONE'
   if paths.filep(vocab_rep_path) then
     local temp_vec = vigo.load(vocab_rep_path)
     for k,v in pairs(temp_vec) do temp_key = k end
     local temp_table = {}
     table.insert(temp_table,temp_key)
     for k,v in pairs(temp_vec[temp_key]) do table.insert(temp_table,v) end
     vocab_rep = torch.Tensor(temp_table) 
   end
   return out, caption, vocab_rep
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
