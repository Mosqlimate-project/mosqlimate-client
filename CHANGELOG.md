Release Notes
---

# [1.5.0](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.4.0...1.5.0) (2024-08-07)


### Features

* **package:** parse JSON objects to pydantic schemas ([#29](https://github.com/Mosqlimate-project/mosqlimate-client/issues/29)) ([f167280](https://github.com/Mosqlimate-project/mosqlimate-client/commit/f16728056289b5be45d3d663d82c5e6c9b52e2a0))

# [1.4.0](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.3.2...1.4.0) (2024-07-18)


### Features

* **config:** replace env variable by a global setting ([#28](https://github.com/Mosqlimate-project/mosqlimate-client/issues/28)) ([370690e](https://github.com/Mosqlimate-project/mosqlimate-client/commit/370690e0088ec8151337abaa7abf97fb24dae472))

## [1.3.2](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.3.1...1.3.2) (2024-06-25)


### Bug Fixes

* **update_py_version:** update python version to <3.13 ([5bcaa23](https://github.com/Mosqlimate-project/mosqlimate-client/commit/5bcaa23e059cc34524db1fc555b5d02cf8d0f2b4))

## [1.3.1](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.3.0...1.3.1) (2024-06-25)


### Bug Fixes

* **post:** restrain prediction data to be a pure json only ([#25](https://github.com/Mosqlimate-project/mosqlimate-client/issues/25)) ([513dec0](https://github.com/Mosqlimate-project/mosqlimate-client/commit/513dec040457d71d9772d12f375e5aff11c771f6))

# [1.3.0](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.2.3...1.3.0) (2024-06-24)


### Bug Fixes

* **dependencies:** ignore jinja2 problem ([#23](https://github.com/Mosqlimate-project/mosqlimate-client/issues/23)) ([836ade6](https://github.com/Mosqlimate-project/mosqlimate-client/commit/836ade6bf9507fe8f9856fd3b4ad22a935315cdb))
* **post:** fix minor errors in post method ([f3a8616](https://github.com/Mosqlimate-project/mosqlimate-client/commit/f3a861649ef249db649c9d6d4c4b8e9a6ed7bf53))
* **release:** mirror python dependency on conda env & poetry ([#24](https://github.com/Mosqlimate-project/mosqlimate-client/issues/24)) ([9269b3d](https://github.com/Mosqlimate-project/mosqlimate-client/commit/9269b3de933d8e4d066724e4f2c892acc333b1e7))


### Features

* **score:** Add first score module ([0a9c8ed](https://github.com/Mosqlimate-project/mosqlimate-client/commit/0a9c8edd4bd2be791617c95415b65839971c3d42))

## [1.2.3](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.2.2...1.2.3) (2024-06-19)


### Bug Fixes

* **Prediction:** update post_prediction types to accept only JSON serializable objects ([#21](https://github.com/Mosqlimate-project/mosqlimate-client/issues/21)) ([d286b40](https://github.com/Mosqlimate-project/mosqlimate-client/commit/d286b40cb9b5c38d1fd42a9466a4a09809376923))

## [1.2.2](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.2.1...1.2.2) (2024-06-17)


### Bug Fixes

* **requests:** include first page on results ([#19](https://github.com/Mosqlimate-project/mosqlimate-client/issues/19)) ([4d6062b](https://github.com/Mosqlimate-project/mosqlimate-client/commit/4d6062bcca12d1d13cfbeefa8cc752d4c88f7dbb))

## [1.2.1](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.2.0...1.2.1) (2024-06-16)


### Bug Fixes

* **baseline:** remove data loader from arima class ([3868fd3](https://github.com/Mosqlimate-project/mosqlimate-client/commit/3868fd3c2becbbc8b77d690b0392a0032a9a4db3))
* **datastore:** implement pydantic; improve code; improve imports; async vs threads benchmark; high level functions ([#17](https://github.com/Mosqlimate-project/mosqlimate-client/issues/17)) ([7ca74a9](https://github.com/Mosqlimate-project/mosqlimate-client/commit/7ca74a9e535d821120bc95336fbb701bf1eb7be5))

# [1.2.0](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.1.0...1.2.0) (2024-06-11)


### Bug Fixes

* **registry:** create pydantic type validations; improve registry endpoint; improve struct; include registry.Prediction; include update_model to registry.Model ([#9](https://github.com/Mosqlimate-project/mosqlimate-client/issues/9)) ([7768ee5](https://github.com/Mosqlimate-project/mosqlimate-client/commit/7768ee5ae61d0683f612d6ecdd0e675595e1d91f))


### Features

* **baseline:** first baseline model ([973c2d0](https://github.com/Mosqlimate-project/mosqlimate-client/commit/973c2d0c3d5c56b97971b3f737279a9e4cd69864))

# [1.1.0](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.0.1...1.1.0) (2024-05-28)


### Features

* **datastore:** Add functions to get the data from the datastore ([#8](https://github.com/Mosqlimate-project/mosqlimate-client/issues/8)) ([8573932](https://github.com/Mosqlimate-project/mosqlimate-client/commit/857393242b6b35a915476c1984a38426ab6ab8be))

## [1.0.1](https://github.com/Mosqlimate-project/mosqlimate-client/compare/1.0.0...1.0.1) (2024-05-27)


### Bug Fixes

* **package:** make params parse run after params type checker ([#7](https://github.com/Mosqlimate-project/mosqlimate-client/issues/7)) ([e4d5437](https://github.com/Mosqlimate-project/mosqlimate-client/commit/e4d54370648c8c14ced17be24cad5ef07bc0ce7a))

# 1.0.0 (2024-05-27)


### Bug Fixes

* **package:** add option to change between prod and env environments ([#6](https://github.com/Mosqlimate-project/mosqlimate-client/issues/6)) ([86796ed](https://github.com/Mosqlimate-project/mosqlimate-client/commit/86796ed8c1b370f9f0a1aec977b7eb332aedb02a))

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
