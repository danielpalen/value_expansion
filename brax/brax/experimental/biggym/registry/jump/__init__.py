# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example: an existing Env + a new reward."""
AUTHORS = ('Hiroki Furuta', 'Shixiang Shane Gu')
CONTACTS = ('furuta@weblab.t.u-tokyo.ac.jp', 'shanegu@google.com')
AFFILIATIONS = ('u-tokyo.ac.jp', 'google.com')
DESCRIPTIONS = (
    'HalfCheetah running and jumping',
    'a cheetah hopping',
)

ENVS = dict(
    cheetah=dict(
        module='cheetah:JumpCheetah',
        tracks=('rl',),
    ),)

COMPONENTS = None
