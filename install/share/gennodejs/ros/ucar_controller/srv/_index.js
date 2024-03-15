
"use strict";

let SetLEDMode = require('./SetLEDMode.js')
let GetBatteryInfo = require('./GetBatteryInfo.js')
let SetSensorTF = require('./SetSensorTF.js')
let GetSensorTF = require('./GetSensorTF.js')
let SetMaxVel = require('./SetMaxVel.js')
let GetMaxVel = require('./GetMaxVel.js')

module.exports = {
  SetLEDMode: SetLEDMode,
  GetBatteryInfo: GetBatteryInfo,
  SetSensorTF: SetSensorTF,
  GetSensorTF: GetSensorTF,
  SetMaxVel: SetMaxVel,
  GetMaxVel: GetMaxVel,
};
